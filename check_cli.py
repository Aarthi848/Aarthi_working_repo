import asyncio
import os
import uuid
import json
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
import requests
from langchain_fireworks import ChatFireworks
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore
from rich.console import Console
from typing_extensions import Annotated
from typing import List, Dict, Any, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph_bigtool.utils import convert_positional_only_function_to_tool
from mcp_server_manager import MCPServerManager, MCPServerConfig
from fixer_graph import create_agent
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR) 


MAX_HISTORY = 8

# ─────────────────────────────────────────────────
#  Required credential fields per server
# ─────────────────────────────────────────────────
REQUIRED_CREDENTIAL_FIELDS = {
    "Gitlab": ["GITLAB_URL", "GITLAB_TOKEN", "TOOL_EMAIL"],
    "Github": ["GITHUB_USERNAME", "GITHUB_TOKEN"],
    "Jira": ["JIRA_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"],
    "Jenkins": ["JENKINS_URL", "JENKINS_USERNAME", "JENKINS_API_TOKEN"],
    "Jfrog Artifactory": ["ARTIFACTORY_URL", "ARTIFACTORY_USERNAME", "ARTIFACTORY_API_KEY"],
    "Servicenow": ["SERVICENOW_INSTANCE_URL", "SERVICENOW_USERNAME", "SERVICENOW_PASSWORD"],
    "Prometheus & Grafana":["PROMETHEUS_URL","PROMETHEUS_MCP_SERVER_TRANSPORT","PROMETHEUS_MCP_BIND_HOST","PROMETHEUS_MCP_BIND_PORT",
                                "GRAFANA_URL", "GRAFANA_API_KEY", "GRAFANA_USERNAME", "GRAFANA_PASSWORD"]

}



loaded_credentials: Dict[str, Dict[str, str]] = {}

# ─────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────
API_BASE = "http://10.0.1.149:8283"


console = Console()
load_dotenv()


# ======================
#  Configuration Classes
# ======================
@dataclass
class MCPServerConfig:
    name: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    transport: str = "stdio"
    headers: Optional[Dict[str, str]] = None


# ======================
#  LangGraph Agent with Remote Tools
# ======================
class RemoteLangGraphAgent:
    def __init__(self, tools: Dict[str, Any], selected_servers: List[str]):
        self.tool_registry = tools
        self.selected_servers = selected_servers
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.store = self._initialize_store()
        self.llm = self._get_llm()
        self.graph = None
        self.current_state = None

    def _initialize_store(self) -> InMemoryStore:
        store = InMemoryStore(
            index={
                "embed": self.embeddings,
                "dims": 384,
                "fields": ["description"],
            }
        )
        
            
        for tool_name, tool in self.tool_registry.items():

            # HARD PATCH: Block invalid hallucinated tool
            if tool_name.lower().strip() == "find_similar_incidents_tool":
                print("🔥 Removed invalid tool from registry: find_similar_incidents_tool")
                continue

            desc = getattr(tool, "description", "No description")
            store.put(("tools",), tool_name, {"description": f"{tool_name}: {desc}"})
    
        
        
        # Add math tools
        import math
        import types
        for func_name in dir(math):
            func = getattr(math, func_name)
            if isinstance(func, types.BuiltinFunctionType):
                if lc_tool := convert_positional_only_function_to_tool(func):
                    tool_id = str(uuid.uuid4())
                    self.tool_registry[tool_id] = lc_tool
                    store.put(
                        ("tools",),
                        tool_id,
                        {"description": f"{lc_tool.name}: {lc_tool.description}"}
                    )

        return store

    def _get_llm(self):
        return ChatFireworks(
            # model="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
            # model= "accounts/fireworks/models/kimi-k2-instruct-0905",
            # model= "accounts/fireworks/models/glm-4p6",
            model= "accounts/fireworks/models/deepseek-v3p2",
            # model ="accounts/fireworks/models/llama4-maverick-instruct-basic",
            # model="accounts/fireworks/models/qwen3-vl-235b-a22b-thinking",
            max_tokens=32000,
            # temperature=0.7
        )
  
    
    
    # def _retrieve_tools_function(
    #     self,
    #     query: str,
    #     *,
    #     store: Annotated[BaseStore, InjectedStore],
    # ) -> List[str]:
    #     results = store.search(("tools",), query=query, limit=15)
    #     # :white_tick: ONLY keep tools that belong to selected_servers
    #     filtered = []
    #     for result in results:
    #         tool_id = result.key
    #         # Check if tool name starts with any selected server (case-insensitive)
    #         if any(tool_id.lower().startswith(s.lower()) for s in self.selected_servers):
    #             filtered.append(tool_id)
    #         if len(filtered) >= 5:
    #             break
    #     return filtered
    
    
    
    def _retrieve_tools_function(
        self,
        query: str,
        *,
        store: Annotated[BaseStore, InjectedStore],
    ) -> List[str]:
        """
        Retrieve relevant tools from the vector store and filter them
        based on selected servers.

        This version NORMALIZES names to safely handle:
        - spaces vs underscores
        - case differences
        - multi-word server names (e.g., 'Jfrog Artifactory')
        """

        def normalize(name: str) -> str:
            return name.lower().replace(" ", "_")

        # 🔍 Search tools semantically
        results = store.search(("tools",), query=query, limit=15)

        filtered: List[str] = []

        for result in results:
            tool_id = result.key  # actual MCP tool name

            normalized_tool = normalize(tool_id)

            # ✅ Match tool to any selected server safely
            for server in self.selected_servers:
                normalized_server = normalize(server)

                if normalized_tool.startswith(normalized_server):
                    filtered.append(tool_id)
                    break  # stop checking other servers for this tool

            # Limit number of tools returned
            if len(filtered) >= 5:
                break

        return filtered

    


    async def _retrieve_tools_coroutine(
        self,
        query: str,
        *,
        store: Annotated[BaseStore, InjectedStore],
    ):
        return self._retrieve_tools_function(query, store=store)
    

    def initialize_agent(self):
        builder = create_agent(
            self.llm,
            self.tool_registry,
            retrieve_tools_function=self._retrieve_tools_function,
            retrieve_tools_coroutine=self._retrieve_tools_coroutine,
            limit=5,
            active_servers=", ".join(self.selected_servers)
        )
        # self.graph = builder.compile(store=self.store, debug=False)
        self.graph = builder.compile(store=self.store, debug=False)

        return self.graph

    def reset_conversation(self):
        self.current_state = None
        console.print("[yellow]🔄 Conversation state reset[/yellow]")

    # def _is_prometheus_yaml_command(self, query: str) -> bool:
    #     """Check if query is specifically for Prometheus YAML management."""
    #     q = query.lower()
    #     yaml_keywords = ["add job", "remove job", "delete job", "add target", 
    #                     "remove target", "list jobs", "list targets", "reload prometheus"]
    #     return any(keyword in q for keyword in yaml_keywords)

  
    
    
    async def process_query(self, query: str) -> str:
        if not query or not query.strip():
            return "Please enter a query to proceed."
        console.print(f"[blue]:magnifying_glass: Processing query: {query}[/blue]")
        
        
        if self.current_state is None:
            self.current_state = {
                "messages": [HumanMessage(content=query)],
                "selected_tool_ids": [],
                "tool_executed": False,
                "retry_count": 0  
            }
        # else:
        #     self.current_state["messages"].append(HumanMessage(content=query))
        #     self.current_state["messages"] = self.current_state["messages"][-MAX_HISTORY:]
        #     print(len(self.current_state["messages"]))
        #     console.print(f"[violet][DEBUG] Adding user query to state: {query}[/violet]")

        #     self.current_state["tool_executed"] = False
            
            
        else:
            self.current_state["messages"].append(HumanMessage(content=query))
            self.current_state["messages"] = self.current_state["messages"][-MAX_HISTORY:]

            # ✅ RESET PER QUERY
            self.current_state["selected_tool_ids"] = []
            self.current_state["tool_executed"] = False
            self.current_state["retry_count"] = 0

            # 🔍 DEBUG (THIS LINE)
            console.print(
                f"[cyan][DEBUG] selected_tool_ids (after reset) = {self.current_state['selected_tool_ids']}[/cyan]"
            )

            console.print(f"[violet][DEBUG] Adding user query to state: {query}[/violet]")
    
        
        try:
            last_ai_response = None
            # We no longer prioritize raw tool output for final response
            # (Formatting is handled in the 'output' node)
            async for step in self.graph.astream(
                self.current_state,
                stream_mode="updates",
                config={"recursion_limit": 30}
            ):
                for _, update in step.items():
                    if "messages" in update:
                        self.current_state["messages"].extend(update["messages"])
                    for msg in update.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            console.print("[yellow]:robot_face: AI Tool Calls:[/yellow]")
                            for tc in msg.tool_calls:
                                console.print(f"  {tc['name']}: {tc['args']}")
                        elif isinstance(msg, ToolMessage):
                            console.print(f"[green]:hammer_and_spanner: Tool Result: {msg.content[:200]}...[/green]")
                            # We log tool results, but do NOT use them as final output
                        elif isinstance(msg, AIMessage) and not msg.tool_calls:
                            console.print(f"[cyan]:bulb: AI Response: {msg.content[:200]}...[/cyan]")
                            # This will be the clean, formatted message from the 'output' node
                            last_ai_response = msg.content
            # :white_tick: Return ONLY the final AI response (from 'output' node)
            if last_ai_response:
                console.print("[violet][DEBUG][BEFORE RESULT] Preparing to return final response[/violet]")
                console.print(f"[DEBUG][BEFORE RESULT] Final response preview: {last_ai_response[:300]}")
                return last_ai_response.strip()
            else:
                # Fallback: only if the graph never produced a final message
                return "Sorry, I couldn't generate a response. Please try again."
        except Exception as e:
            error_text = str(e).lower()
            if "recursion limit" in error_text:
                
                
                #added new
                logger.error("Recursion detected", exc_info=True)
                console.print("[red]:x: Internal Recursion Error (hidden from user)[/red]")
                return "Something went wrong. Please try again."
            if "json schema" in error_text or "invalid_request_error" in error_text:
                console.print(f"[red]:x: 3rd Party Tool Error: {str(e)}[/red]")
                return (
                    "The external tool rejected the request. "
                    "This tool may require specific parameters. "
                    "Please try rephrasing your query or ask what information the tool needs."
                )
            console.print(f"[red]:x: Internal Error: {str(e)}[/red]")
            return "Something went wrong while processing your request. Please try again."



# ======================
#  Credential Fetcher
# ======================
def fetch_user_credentials(user_id: int, server_name: str) -> Dict[str, str]:
    if server_name in loaded_credentials:
        return loaded_credentials[server_name]

    url = f"{API_BASE}/credentials/{user_id}/{server_name}"
    res = requests.get(url, timeout=10)

    if res.status_code == 200:
        creds = res.json().get("credentials", {})
        loaded_credentials[server_name] = creds
        console.print(f"[green]✓ Loaded credentials for {server_name} (User {user_id})[/green]")
        return creds

    console.print(f"[yellow]⚠ No credentials for {server_name} (User {user_id})[/yellow]")
    return {}



# ======================
#  Main Function
# ======================
async def main():
    db_config = {
        "host": "10.0.1.39",
        "user": "resonance",
        "password": "VBhkk!op",
        "database": "SSOauthentication"
    }
    print(f"[DEBUG] db config loaded")
    server_manager = MCPServerManager(db_config)

    os.environ["FIREWORKS_API_KEY"] = "fw_3ZPF2zBvheEvfSGRRoChVZPc"
    console.print("[blue]🚀 Starting MCP client with remote support...[/blue]")

    if not server_manager.servers:
        console.print("[yellow]⚠ No servers configured in mcp_servers.json[/yellow]")
        return

    email = input("👤 Enter your email: ").strip()
    try:
        res = requests.get(f"{API_BASE}/users/lookup", params={"email": email}, timeout=10)
        res.raise_for_status()
        user_id = res.json()["user_id"]
        console.print(f"[green]✅ Welcome back! (User ID: {user_id})[/green]")
        console.print(f"[pink] login successful[/pink]")
    except requests.exceptions.RequestException as e:
        console.print(f"[red]❌ Failed to fetch user ID: {e}[/red]")
        return

   
    # -----------------------------------------
    #  FETCH ENTERPRISE 3rd-PARTY TOOLS (FIXED)
    # -----------------------------------------
    def fetch_enterprise_tools(user_id: int, email: str):
        # url = f"{API_BASE}/enterprise-configuration?user_id={user_id}&user_email={email}"
        url = f"{API_BASE}/devtools/enterprise?user_id={user_id}&user_email={email}"
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            console.print("[yellow]⚠ Could not load enterprise tools[/yellow]")
            return {}
        print(res.json)
        return res.json()



    console.print("[cyan]🔗 Loading enterprise tools from API...[/cyan]")
    enterprise_data = fetch_enterprise_tools(user_id, email)

    enterprise_tools = enterprise_data.get("tools", {})

    added_tools = 0

    for tool_name, info in enterprise_tools.items():
        # Skip built-in tools (they are already handled)
        if tool_name in REQUIRED_CREDENTIAL_FIELDS:
            continue

        saved = info.get("saved_configuration", {})
        
        #  Get transport type from API response
        # transport = saved.get("transport", "http")
        transport = saved.get("transport") or info.get("transport") or "http"
        
        
        # ═══════════════════════════════════════════
        #  STDIO TRANSPORT
        # ═══════════════════════════════════════════
        if transport == "stdio":
            auth = saved.get("auth", {})
            
            if not auth:
                console.print(f"[yellow]⚠ Missing auth config for STDIO tool: {tool_name}[/yellow]")
                continue
            
            # Extract from headers JSON or direct fields
            headers = auth.get("headers", {})
            
            if isinstance(headers, str):
                import json
                headers = json.loads(headers)
            
            # REQUIRED
            command = auth.get("username") or headers.get("COMMAND")
            args = headers.get("ARGS") or []                  # ← FIXED (list)
            env_vars = headers.get("ENV", {})                 # ← FIXED (added)
            working_dir = headers.get("WORKING_DIR", "")      # optional
            
            if not command:
                console.print(f"[yellow]⚠ Missing COMMAND for STDIO tool: {tool_name}[/yellow]")
                continue
            
            # Register STDIO tool correctly
            server_manager.servers[tool_name] = MCPServerConfig(
                name=tool_name,
                command=command,
                args=args,
                transport="stdio",
                headers=env_vars    # ← IMPORTANT: pass env
            )
            
            console.print(f"[green]✓ Loaded STDIO tool: {tool_name} → {command}[/green]")
            added_tools += 1

        
        # ═══════════════════════════════════════════
        #  HTTP TRANSPORT
        # ═══════════════════════════════════════════
        else:
            # base_url = saved.get("base_url") or saved.get("BASE_URL") or saved.get("url")
            base_url = (
                saved.get("server_url") or
                saved.get("base_url") or
                saved.get("BASE_URL") or
                saved.get("url")
            )

            
            if not base_url:
                console.print(f"[yellow]⚠ Missing base_url for HTTP tool: {tool_name}[/yellow]")
                continue
            
            #  Register HTTP tool
            server_manager.servers[tool_name] = MCPServerConfig(
                name=tool_name,
                url=base_url,
                transport="streamable_http"
            )
            
            console.print(f"[green]✓ Loaded HTTP tool: {tool_name} → {base_url}[/green]")
            added_tools += 1

    console.print(f"[green]✓ Loaded {added_tools} enterprise tools[/green]")

    console.print("[red]♻ Restarting MCP servers with new user credentials...[/red]")

    console.print("\n[bold]Available Servers (from DB):[/bold]")
    for i, (name, server) in enumerate(server_manager.servers.items(), 1):
        label = f"{i}. {name}"
        if server.url:
            console.print(f"{label} → {server.url} ({server.transport})")
        else:
            console.print(f"{label} → {server.command} {' '.join(server.args or [])}")

    selection = input("\n Select servers (e.g. 1,2) or 'all': ").strip()
    selected = list(server_manager.servers.keys()) if selection.lower() == "all" else [
        list(server_manager.servers.keys())[int(idx.strip()) - 1]
        for idx in selection.split(",") if idx.strip().isdigit()
    ]

    if not selected:
        console.print("[red] No servers selected[/red]")
        return

    mcp_client_config = {}
    for name in selected:
        server = server_manager.servers[name]
        if server.url:
            mcp_client_config[name] = {
                "transport": server.transport or "streamable_http",
                "url": server.url,
                "headers": {}
            }
        else:
            creds = fetch_user_credentials(user_id, name)
            required = REQUIRED_CREDENTIAL_FIELDS.get(name, [])
            missing = [k for k in required if k not in creds]
            


            if missing:
                console.print(f"[red] Missing for {name}: {missing}[/red]")
                continue

            env_creds = {k: creds[k] for k in required}
            
            # Combine fetched user credentials with any custom env vars from server.headers (enterprise config)
            # User credentials (env_creds) should typically take precedence.
            final_env = {**(server.headers or {}), **env_creds}

            mcp_client_config[name] = {
                "transport": server.transport or "stdio",
                "command": server.command,
                "args": server.args or [],
                "env": final_env, # <-- CORRECTLY passing combined environment dictionary
            }

    console.print("[cyan] Connecting to servers via MultiServerMCPClient...[/cyan]")
    try:
        client = MultiServerMCPClient(mcp_client_config)
        tools = await client.get_tools()
        tool_dict = {tool.name: tool for tool in tools}
        console.print(f"[green]✓ Loaded {len(tools)} tools from {len(selected)} servers[/green]")
        
        # # Show available Prometheus tools
        # prom_tools = [t for t in tool_dict.keys() if "prometheus" in t.lower() or "metric" in t.lower()]
        # if prom_tools:
        #     console.print(f"[cyan] Prometheus tools available: {', '.join(prom_tools)}[/cyan]")
    except Exception as e:
        console.print(f"[red] Failed to connect: {e}[/red]")
        return

    console.print("[blue]⏳ Initializing LangGraph Agent...[/blue]")
    agent = RemoteLangGraphAgent(tool_dict, selected)
    agent.initialize_agent()
    console.print("[bold green] Agent Ready![/bold green] Type 'exit' to quit, 'reset' to clear conversation.")

    while True:
        user_query = input("\n🔍 Query: ").strip()
        if user_query.lower() in ('exit', 'quit'):
            break
        elif user_query.lower() == 'reset':
            agent.reset_conversation()
            continue

        # # ⚙️ Handle ONLY explicit Prometheus YAML commands locally
        # if agent._is_prometheus_yaml_command(user_query):
        #     console.print("[yellow] Detected Prometheus YAML management command...[/yellow]")
        #     try:
        #         response = process_natural_query(user_query)
        #         console.print(f"[green] YAML Update Result:[/green] {response}")
        #         continue
        #     except Exception as e:
        #         console.print(f"[red] YAML operation failed: {e}[/red]")
        #         continue

        # Otherwise use MCP + LangGraph agent for ALL other queries
        result = await agent.process_query(user_query)
        console.print(f"\n[green] Final Result:[/green]\n{result}")
        console.print("[violet][DEBUG][AFTER RESULT] Result received from agent[/violet]")

    try:
        await client.aclose()
        console.print("[violet]all resources are closed[/violet]")
    except:
        
        await client.aclose()
        console.print("[violet]exception occur - all resources are not closed[/violet]")
        pass


if __name__ == "__main__":
    asyncio.run(main())




