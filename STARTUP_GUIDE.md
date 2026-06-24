# How to Start the EnergyPlus MCP Agent

Whenever you open this project in VSCode, open a new terminal and run the following commands:

```bash
# 1. Activate the isolated Python environment
source venv/bin/activate

# 2. Start the Agent server
python agent.py
```

Once you see the `Agent ready ✔` message in the terminal, open your web browser and go to:
**http://localhost:5000**

## Troubleshooting
- **Port 5000 busy?** The agent will automatically try to find the next available port (e.g., 5001, 5002). Just check the terminal output for the correct URL.
- **Docker permission denied?** If your computer restarts and you get a Docker permission error, run `newgrp docker` in your terminal before starting the agent.
