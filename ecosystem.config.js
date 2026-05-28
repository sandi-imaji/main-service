// Ecosystem PM2 untuk Main Service
// Membaca konfigurasi dari .env.services di root directory

const fs = require('fs');
const path = require('path');

// Load environment variables from .env.services
function loadEnvServices() {
  const rootDir = path.join(__dirname, '..');
  const envPath = path.join(rootDir, '.env.services');
  
  if (fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, 'utf8');
    const lines = envContent.split('\n');
    
    lines.forEach(line => {
      // Skip comments and empty lines
      if (line.trim().startsWith('#') || !line.trim()) return;
      
      const match = line.match(/^([^=]+)=(.*)$/);
      if (match) {
        const key = match[1].trim();
        const value = match[2].trim();
        process.env[key] = value;
      }
    });
  }
}

// Load env
loadEnvServices();

module.exports = {
  apps: [
    {
      name: "main-service",
      cwd: __dirname,
      script: process.env.MAIN_SERVICE_VENV || "/home/imaji/smart-ai/smart-ai-dev/main-service/.venv/bin/python",
      args: ["DEBUG=1",
        "-m", "uvicorn", 
        "app.server:app", 
        "--host", process.env.MAIN_SERVICE_HOST || "0.0.0.0", 
        "--port", process.env.MAIN_SERVICE_PORT || "8080"
      ],
      exec_mode: "fork",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        NODE_ENV: "production",
        DEBUGMODE: "false",
        HOST_MAIN: process.env.MAIN_SERVICE_HOST || "0.0.0.0",
        PORT_MAIN: process.env.MAIN_SERVICE_PORT || "8080"
      },
      env_development: {
        NODE_ENV: "development",
        DEBUGMODE: "true",
        HOST_MAIN: process.env.MAIN_SERVICE_HOST || "0.0.0.0",
        PORT_MAIN: process.env.MAIN_SERVICE_PORT || "8080"
      },
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      time: true
    }
  ]
};
