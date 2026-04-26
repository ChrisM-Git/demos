#!/bin/bash

###############################################################################
# START ALL MCP SERVERS + MULTI-AGENT DEMO
###############################################################################

echo "═══════════════════════════════════════════════════════════════════"
echo "   STARTING MULTI-AGENT DEMO WITH ALL MCP SERVERS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Check if API key is set
if [ -z "$PANW_AI_SEC_API_KEY" ]; then
    echo "⚠️  WARNING: PANW_AI_SEC_API_KEY not set"
    echo "   Please run: export PANW_AI_SEC_API_KEY=your_key_here"
    echo ""
fi

# Create logs directory
mkdir -p logs

echo "🚀 Starting MCP Servers..."
echo ""

# Start all MCP servers in background
echo "  📦 Starting customer-service MCP server (port 9001)..."
python3 customer_mcp_server.py > logs/customer_mcp.log 2>&1 &
CUST_PID=$!
sleep 1

echo "  📦 Starting order-history MCP server (port 9002)..."
python3 order_history_mcp_server.py > logs/order_history_mcp.log 2>&1 &
ORDER_PID=$!
sleep 1

echo "  🏪 Starting product-catalog MCP server (port 9003)..."
python3 product_catalog_mcp_server.py > logs/product_catalog_mcp.log 2>&1 &
PRODUCT_PID=$!
sleep 1

echo "  🤖 Starting ml-recommendation MCP server (port 9004)..."
python3 ml_recommendation_mcp_server.py > logs/ml_recommendation_mcp.log 2>&1 &
ML_PID=$!
sleep 1

echo "  📊 Starting analytics MCP server (port 9005)..."
python3 analytics_mcp_server.py > logs/analytics_mcp.log 2>&1 &
ANALYTICS_PID=$!
sleep 1

echo "  📬 Starting notification MCP server (port 9006)..."
python3 notification_mcp_server.py > logs/notification_mcp.log 2>&1 &
NOTIF_PID=$!
sleep 1

echo ""
echo "✅ All MCP servers started!"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "   MCP SERVERS RUNNING:"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  🏢 Customer Service:     http://localhost:9001  (PID: $CUST_PID)"
echo "  📦 Order History:        http://localhost:9002  (PID: $ORDER_PID)"
echo "  🏪 Product Catalog:      http://localhost:9003  (PID: $PRODUCT_PID)"
echo "  🤖 ML Recommendations:   http://localhost:9004  (PID: $ML_PID)"
echo "  📊 Analytics:            http://localhost:9005  (PID: $ANALYTICS_PID)"
echo "  📬 Notifications:        http://localhost:9006  (PID: $NOTIF_PID)"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Wait for servers to be ready
echo "⏳ Waiting for MCP servers to be ready..."
sleep 3

# Health check
echo ""
echo "🏥 Health Check:"
for port in 9001 9002 9003 9004 9005 9006; do
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo "  ✅ Port $port - OK"
    else
        echo "  ❌ Port $port - NOT RESPONDING"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "   STARTING MULTI-AGENT ORCHESTRATOR"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Start the main multi-agent demo
echo "🤖 Starting Multi-Agent Demo (port 8079)..."
echo ""
python3 multi_agent_demo.py > logs/multi_agent_demo.log 2>&1 &
AGENT_PID=$!

sleep 2

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "   ✅ ALL SYSTEMS RUNNING!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📡 Multi-Agent Demo:     http://localhost:8079"
echo "🌐 Web Interface:        multi_agent_chat.html"
echo "📊 Landing Page:         landpage.html → Multi-Agent Recommendations"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Process IDs:"
echo "  Agent Orchestrator: $AGENT_PID"
echo "  MCP Servers: $CUST_PID $ORDER_PID $ANALYTICS_PID $ML_PID $PRODUCT_PID $NOTIF_PID"
echo ""
echo "Logs:"
echo "  All logs are in the logs/ directory"
echo "  tail -f logs/multi_agent_demo.log  # Watch agent"
echo "  tail -f logs/customer_mcp.log      # Watch customer service"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "To stop all servers:"
echo "  kill $AGENT_PID $CUST_PID $ORDER_PID $ANALYTICS_PID $ML_PID $PRODUCT_PID $NOTIF_PID"
echo ""
echo "Or run: ./stop_all.sh"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "🎉 Ready to demo! Open multi_agent_chat.html in your browser."
echo ""

# Save PIDs to file for stop script
echo "$AGENT_PID $CUST_PID $ORDER_PID $ANALYTICS_PID $ML_PID $PRODUCT_PID $NOTIF_PID" > .pids

# Keep script running
wait $AGENT_PID