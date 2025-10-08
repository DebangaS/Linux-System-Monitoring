"""
API Documentation generator
Author: Member 1
"""

from flask import Blueprint, jsonify, render_template_string
from datetime import datetime

docs_bp = Blueprint('docs', __name__, url_prefix='/api')

API_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "System Monitor API",
        "version": "1.0.0",
        "description": "Real-time system resource monitoring API",
        "contact": {
            "name": "System Monitor Team",
            "email": "support@systemmonitor.com"
        }
    },
    "servers": [
        {
            "url": "/api/v1",
            "description": "Production server"
        }
    ],
    "paths": {
        "/health": {
            "get": {
                "summary": "Health check endpoint",
                "description": "Returns the health status of the API",
                "responses": {
                    "200": {
                        "description": "Service is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "timestamp": {"type": "string"},
                                        "version": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/system/resources": {
            "get": {
                "summary": "Get current system resources",
                "description": "Returns current CPU, memory, and disk usage",
                "responses": {
                    "200": {
                        "description": "System resources data",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "cpu": {
                                            "type": "object",
                                            "properties": {
                                                "usage_percent": {"type": "number"},
                                                "per_core": {
                                                    "type": "array",
                                                    "items": {"type": "number"}
                                                }
                                            }
                                        },
                                        "memory": {
                                            "type": "object",
                                            "properties": {
                                                "total": {"type": "integer"},
                                                "available": {"type": "integer"},
                                                "used": {"type": "integer"},
                                                "percent": {"type": "number"}
                                            }
                                        },
                                        "disk": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "device": {"type": "string"},
                                                    "mountpoint": {"type": "string"},
                                                    "total": {"type": "integer"},
                                                    "used": {"type": "integer"},
                                                    "free": {"type": "integer"},
                                                    "percent": {"type": "number"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/system/processes": {
            "get": {
                "summary": "Get running processes",
                "description": "Returns list of running processes with filtering options",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of processes to return",
                        "schema": {"type": "integer", "default": 50}
                    },
                    {
                        "name": "sort_by",
                        "in": "query",
                        "description": "Sort processes by field",
                        "schema": {
                            "type": "string",
                            "enum": ["cpu_percent", "memory_percent", "name"],
                            "default": "cpu_percent"
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "List of processes",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "processes": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "pid": {"type": "integer"},
                                                    "name": {"type": "string"},
                                                    "username": {"type": "string"},
                                                    "cpu_percent": {"type": "number"},
                                                    "memory_percent": {"type": "number"},
                                                    "status": {"type": "string"}
                                                }
                                            }
                                        },
                                        "total_count": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "components": {
        "securitySchemes": {
            "sessionAuth": {
                "type": "apiKey",
                "in": "cookie",
                "name": "session"
            }
        }
    }
}

# Fixed HTML for Swagger UI
SWAGGER_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>System Monitor API Documentation</title>
    <link rel="stylesheet" type="text/css"
          href="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui.css">
    <style>
        html { box-sizing: border-box; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin: 0; background: #fafafa; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            SwaggerUIBundle({
                url: '/api/spec',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            });
        }
    </script>
</body>
</html>
"""

@docs_bp.route('/docs')
def api_docs():
    """Render API documentation UI"""
    return render_template_string(SWAGGER_HTML)

@docs_bp.route('/spec')
def api_spec():
    """Return OpenAPI specification"""
    return jsonify(API_SPEC)

@docs_bp.route('/info')
def api_info():
    """Return API information"""
    return jsonify({
        "name": "System Monitor API",
        "version": "1.0.0",
        "description": "Real-time system resource monitoring API",
        "documentation_url": "/api/docs",
        "specification_url": "/api/spec",
        "last_updated": datetime.utcnow().isoformat(),
        "endpoints": {
            "health": "/api/v1/health",
            "system_resources": "/api/v1/system/resources",
            "processes": "/api/v1/system/processes",
            "authentication": "/auth/login"
        },
        "rate_limits": {
            "api": "100 requests per minute",
            "auth": "5 requests per minute"
        }
    })
