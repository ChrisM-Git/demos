# Multi-Agent Demo with Real MCP Integration

## What Changed

Your `multi_agent_demo.py` has been updated to use **REAL MCP server calls** instead of mocked responses!

### Before:
- All MCP responses were hardcoded in `_get_mock_mcp_response()`
- No actual HTTP calls to MCP servers
- Just simulated delays and fake data

### After:
- Makes real HTTP POST requests to your MCP servers
- Calls endpoints like `http://localhost:9001/mcp/tools/call`
- Falls back to mock data if servers aren't running
- Ready for Palo Alto AIRS proxy integration

## Your API Key

✅ Already configured: `3minTOxgbmsFSceHHqqiX2SaCNveGZKjdHIDkGt7llWMmKwf`

## How to Run (On Your Webserver)

### Step 1: Start All MCP Servers

```bash
cd /path/to/mcp/directory
chmod +x start_all.sh
./start_all.sh
```

This will start:
- Customer Service (port 9001)
- Order History (port 9002)
- Product Catalog (port 9003)
- ML Recommendations (port 9004)
- Analytics (port 9005)
- Notifications (port 9006)
- Multi-Agent Orchestrator (port 8079)

### Step 2: Open the Web Interface

Open `multi_agent_chat.html` in your browser:
```
http://your-webserver-ip:8000/multi_agent_chat.html
```

### Step 3: Test with Real Agents

Try these example goals:
1. **"Recommend products for customer Sarah Chen"**
   - Calls real customer-service MCP server
   - Calls real order-history MCP server
   - Calls real analytics MCP server
   - Calls real ML recommendation MCP server
   - Sends results through notification server

2. **"Analyze sales forecast"** (coming soon)

## What Happens Now

### Real MCP Flow:
```
User Goal
   ↓
Multi-Agent Orchestrator (port 8079)
   ↓
HTTP POST to http://localhost:9001/mcp/tools/call
   {
     "tool": "get_customer_profile",
     "arguments": {"customer_id": "CUST-12345"}
   }
   ↓
Real Customer MCP Server Response
   {
     "success": true,
     "data": { ... actual customer data ... }
   }
   ↓
Agent continues with next step
```

### Security Inspection:
The `_check_security_policy()` method still simulates AIRS security:
- ✅ Allows most calls
- 🛑 Blocks external emails
- 🛑 Blocks bulk exports
- 🛑 Blocks destructive operations

## Verify It's Working

### Check MCP Server Logs:
```bash
tail -f logs/customer_mcp.log
tail -f logs/multi_agent_demo.log
```

You should see:
```
🔧 MCP Tool Call: get_customer_profile
   Arguments: {"customer_id": "CUST-12345"}
👤 Getting profile for CUST-12345
✅ Found customer: Sarah Chen
```

### Check Network Calls:
In the multi-agent demo logs, you'll see:
```
📡 MCP Call [abc123]: customer-service.get_customer_profile
🔍 Calling real MCP server: customer-service...
✅ MCP Response [abc123]: Success
```

## Fallback Behavior

If MCP servers aren't running, the code automatically falls back to mock data with a warning:
```
⚠️ Falling back to mock data for customer-service.get_customer_profile
```

This ensures the demo always works!

## Next Steps: Full AIRS Proxy Integration

To route ALL calls through Palo Alto AIRS MCP Proxy:

1. **Update the MCP endpoint** in `multi_agent_demo.py`:
   ```python
   # Change from direct server calls to AIRS proxy
   self.use_airs_proxy = True
   ```

2. **Configure AIRS proxy URL** - The AIRS proxy will need to know about your MCP servers

3. **Update the call flow**:
   ```
   Agent → AIRS Proxy → MCP Server
            ↑
         Security
         Inspection
   ```

## Testing Real vs Mock Data

To verify you're getting real data:

1. **Modify customer data** in `customer_mcp_server.py`:
   ```python
   CUSTOMERS = {
       "CUST-12345": {
           "name": "REAL SERVER DATA - Sarah Chen",  # Changed!
           ...
       }
   }
   ```

2. **Restart customer MCP server**:
   ```bash
   kill <customer_mcp_pid>
   python3 customer_mcp_server.py > logs/customer_mcp.log 2>&1 &
   ```

3. **Run a test** - You should see "REAL SERVER DATA" in the response!

## Dependencies

Make sure `httpx` is installed:
```bash
pip3 install httpx
```

## Troubleshooting

### "Connection refused" errors:
- Check MCP servers are running: `ps aux | grep mcp_server`
- Check ports: `netstat -an | grep 900[1-6]`

### "Falling back to mock data":
- MCP servers not started
- Wrong ports configured
- Firewall blocking localhost connections

### HTML shows "Could not connect to agent":
- Multi-agent demo not running on port 8079
- Check: `curl http://localhost:8079/health`

## Architecture

```
┌─────────────────────┐
│  Web Browser        │
│  (multi_agent_chat  │
│   .html)            │
└──────────┬──────────┘
           │ HTTP POST /execute-goal
           ↓
┌─────────────────────┐
│  Multi-Agent        │
│  Orchestrator       │
│  (port 8079)        │
└──────────┬──────────┘
           │ Real HTTP Calls
           ↓
┌──────────────────────────────────┐
│  MCP Servers (FastAPI)           │
├──────────────────────────────────┤
│  • Customer Service (9001)       │
│  • Order History (9002)          │
│  • Product Catalog (9003)        │
│  • ML Recommendations (9004)     │
│  • Analytics (9005)              │
│  • Notifications (9006)          │
└──────────────────────────────────┘
```

## Files Modified

- ✅ `multi_agent_demo.py` - Now makes real HTTP calls
- ✅ `start_all.sh` - Fixed port ordering
- ✅ `README_REAL_MCP.md` - This file!

## Original Mock Data Location

The mock data is still available as fallback in:
- `multi_agent_demo.py` → `_get_mock_mcp_response()` method (lines 224-378)

This ensures the demo works even without MCP servers running!
