#!/usr/bin/env python3
"""
API Documentation Generator
Comprehensive documentation system for the holistic diagnostic platform APIs.
Generates interactive documentation, code examples, and integration guides.
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, Template
import inspect
import ast
from enum import Enum
import re
import markdown
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentationType(Enum):
    """Types of documentation to generate."""
    API_REFERENCE = "api_reference"
    USER_GUIDE = "user_guide"
    INTEGRATION_GUIDE = "integration_guide"
    CODE_EXAMPLES = "code_examples"
    OPENAPI_SPEC = "openapi_spec"

class OutputFormat(Enum):
    """Documentation output formats."""
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    JSON = "json"
    YAML = "yaml"

@dataclass
class APIEndpoint:
    """API endpoint documentation structure."""
    path: str
    method: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    security: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert endpoint to dictionary."""
        return {
            "path": self.path,
            "method": self.method,
            "summary": self.summary,
            "description": self.description,
            "parameters": self.parameters,
            "requestBody": self.request_body,
            "responses": self.responses,
            "tags": self.tags,
            "security": self.security,
            "examples": self.examples
        }

@dataclass
class APISchema:
    """API schema/model documentation."""
    name: str
    type: str
    description: str
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    example: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "properties": self.properties,
            "required": self.required,
            "example": self.example
        }

@dataclass
class DocumentationSection:
    """Documentation section structure."""
    title: str
    content: str
    subsections: List['DocumentationSection'] = field(default_factory=list)
    code_examples: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
            "code_examples": self.code_examples
        }

class APIAnalyzer:
    """Analyze source code to extract API documentation."""
    
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.endpoints: List[APIEndpoint] = []
        self.schemas: List[APISchema] = []
        
    def analyze_fastapi_app(self, app_file: str) -> List[APIEndpoint]:
        """Analyze FastAPI application to extract endpoints."""
        endpoints = []
        
        try:
            # Read and parse the FastAPI app file
            with open(app_file, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract API endpoints from decorators
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    endpoint = self._extract_endpoint_from_function(node, content)
                    if endpoint:
                        endpoints.append(endpoint)
            
        except Exception as e:
            logger.error(f"Error analyzing FastAPI app: {str(e)}")
        
        return endpoints
    
    def _extract_endpoint_from_function(self, func_node: ast.FunctionDef, source_content: str) -> Optional[APIEndpoint]:
        """Extract endpoint information from function node."""
        try:
            # Look for FastAPI decorators
            for decorator in func_node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute):
                        method = decorator.func.attr.upper()
                        if method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                            # Extract path
                            path = ""
                            if decorator.args:
                                if isinstance(decorator.args[0], ast.Str):
                                    path = decorator.args[0].s
                                elif isinstance(decorator.args[0], ast.Constant):
                                    path = decorator.args[0].value
                            
                            # Extract docstring
                            docstring = ast.get_docstring(func_node) or ""
                            summary, description = self._parse_docstring(docstring)
                            
                            # Extract function parameters
                            parameters = self._extract_parameters(func_node)
                            
                            # Create endpoint
                            endpoint = APIEndpoint(
                                path=path,
                                method=method,
                                summary=summary,
                                description=description,
                                parameters=parameters,
                                responses=self._generate_default_responses()
                            )
                            
                            return endpoint
            
        except Exception as e:
            logger.warning(f"Error extracting endpoint from function {func_node.name}: {str(e)}")
        
        return None
    
    def _parse_docstring(self, docstring: str) -> tuple[str, str]:
        """Parse function docstring into summary and description."""
        lines = docstring.strip().split('\n')
        if not lines:
            return "", ""
        
        summary = lines[0].strip()
        description = '\n'.join(line.strip() for line in lines[1:]).strip()
        
        return summary, description
    
    def _extract_parameters(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters as API parameters."""
        parameters = []
        
        for arg in func_node.args.args:
            if arg.arg not in ['self', 'request']:  # Skip common non-API parameters
                param = {
                    "name": arg.arg,
                    "in": "query",  # Default to query parameter
                    "required": True,
                    "schema": {"type": "string"}  # Default type
                }
                
                # Try to infer type from annotation
                if arg.annotation:
                    param["schema"] = self._get_type_from_annotation(arg.annotation)
                
                parameters.append(param)
        
        return parameters
    
    def _get_type_from_annotation(self, annotation: ast.expr) -> Dict[str, str]:
        """Get OpenAPI type from Python type annotation."""
        type_mapping = {
            'str': 'string',
            'int': 'integer',
            'float': 'number',
            'bool': 'boolean',
            'list': 'array',
            'dict': 'object'
        }
        
        if isinstance(annotation, ast.Name):
            return {"type": type_mapping.get(annotation.id, "string")}
        elif isinstance(annotation, ast.Constant):
            return {"type": type_mapping.get(type(annotation.value).__name__, "string")}
        
        return {"type": "string"}
    
    def _generate_default_responses(self) -> Dict[str, Dict[str, Any]]:
        """Generate default response documentation."""
        return {
            "200": {
                "description": "Successful response",
                "content": {
                    "application/json": {
                        "schema": {"type": "object"}
                    }
                }
            },
            "400": {
                "description": "Bad request",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "500": {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object", 
                            "properties": {
                                "error": {"type": "string"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    
    def extract_schemas_from_models(self, models_dir: str) -> List[APISchema]:
        """Extract schema documentation from Pydantic models."""
        schemas = []
        models_path = Path(models_dir)
        
        for py_file in models_path.glob("**/*.py"):
            try:
                schemas.extend(self._analyze_models_file(py_file))
            except Exception as e:
                logger.warning(f"Error analyzing models file {py_file}: {str(e)}")
        
        return schemas
    
    def _analyze_models_file(self, file_path: Path) -> List[APISchema]:
        """Analyze a Python file for Pydantic models."""
        schemas = []
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it's a Pydantic model
                if self._is_pydantic_model(node):
                    schema = self._extract_schema_from_class(node)
                    if schema:
                        schemas.append(schema)
        
        return schemas
    
    def _is_pydantic_model(self, class_node: ast.ClassDef) -> bool:
        """Check if class is a Pydantic model."""
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id == 'BaseModel':
                return True
            elif isinstance(base, ast.Attribute) and base.attr == 'BaseModel':
                return True
        return False
    
    def _extract_schema_from_class(self, class_node: ast.ClassDef) -> Optional[APISchema]:
        """Extract schema from Pydantic model class."""
        try:
            name = class_node.name
            docstring = ast.get_docstring(class_node) or f"{name} model"
            
            properties = {}
            required = []
            
            for node in class_node.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    field_name = node.target.id
                    field_type = self._get_type_from_annotation(node.annotation)
                    
                    properties[field_name] = {
                        "type": field_type.get("type", "string"),
                        "description": f"{field_name} field"
                    }
                    
                    # Check if field is required (no default value)
                    if not node.value:
                        required.append(field_name)
            
            return APISchema(
                name=name,
                type="object",
                description=docstring,
                properties=properties,
                required=required
            )
            
        except Exception as e:
            logger.warning(f"Error extracting schema from class {class_node.name}: {str(e)}")
            return None

class DocumentationGenerator:
    """Generate comprehensive API documentation."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir = self.output_dir / "templates"
        self._setup_templates()
        
    def _setup_templates(self):
        """Setup Jinja2 templates for documentation generation."""
        self.templates_dir.mkdir(exist_ok=True)
        
        # HTML template for API documentation
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - API Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 30px; }
        .endpoint { border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; }
        .method { padding: 4px 8px; border-radius: 3px; font-weight: bold; color: white; }
        .get { background-color: #61affe; }
        .post { background-color: #49cc90; }
        .put { background-color: #fca130; }
        .delete { background-color: #f93e3e; }
        .code { background-color: #f4f4f4; padding: 10px; border-radius: 3px; overflow-x: auto; }
        pre { margin: 0; }
        .toc { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
        .toc ul { list-style-type: none; padding-left: 20px; }
        .toc a { text-decoration: none; color: #007bff; }
        .toc a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>{{ description }}</p>
        <p><strong>Version:</strong> {{ version }} | <strong>Generated:</strong> {{ generated_at }}</p>
    </div>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#overview">Overview</a></li>
            <li><a href="#authentication">Authentication</a></li>
            <li><a href="#endpoints">API Endpoints</a>
                <ul>
                    {% for endpoint in endpoints %}
                    <li><a href="#{{ endpoint.path | replace('/', '-') | replace('{', '') | replace('}', '') }}">{{ endpoint.method }} {{ endpoint.path }}</a></li>
                    {% endfor %}
                </ul>
            </li>
            <li><a href="#schemas">Data Schemas</a></li>
            <li><a href="#examples">Code Examples</a></li>
        </ul>
    </div>
    
    <section id="overview">
        <h2>Overview</h2>
        <p>This is the comprehensive API documentation for the Holistic Diagnostic Platform. The API provides endpoints for medical imaging analysis, AI-powered diagnostics, and clinical data management.</p>
    </section>
    
    <section id="authentication">
        <h2>Authentication</h2>
        <p>The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:</p>
        <div class="code">
            <pre>Authorization: Bearer &lt;your-jwt-token&gt;</pre>
        </div>
    </section>
    
    <section id="endpoints">
        <h2>API Endpoints</h2>
        {% for endpoint in endpoints %}
        <div class="endpoint" id="{{ endpoint.path | replace('/', '-') | replace('{', '') | replace('}', '') }}">
            <h3>
                <span class="method {{ endpoint.method.lower() }}">{{ endpoint.method }}</span>
                {{ endpoint.path }}
            </h3>
            <p><strong>Summary:</strong> {{ endpoint.summary }}</p>
            <p>{{ endpoint.description }}</p>
            
            {% if endpoint.parameters %}
            <h4>Parameters</h4>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th style="padding: 8px;">Name</th>
                    <th style="padding: 8px;">Type</th>
                    <th style="padding: 8px;">Required</th>
                    <th style="padding: 8px;">Description</th>
                </tr>
                {% for param in endpoint.parameters %}
                <tr>
                    <td style="padding: 8px;">{{ param.name }}</td>
                    <td style="padding: 8px;">{{ param.schema.type }}</td>
                    <td style="padding: 8px;">{{ 'Yes' if param.required else 'No' }}</td>
                    <td style="padding: 8px;">{{ param.get('description', 'No description') }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
            
            {% if endpoint.responses %}
            <h4>Responses</h4>
            {% for status, response in endpoint.responses.items() %}
            <p><strong>{{ status }}:</strong> {{ response.description }}</p>
            {% endfor %}
            {% endif %}
        </div>
        {% endfor %}
    </section>
    
    <section id="schemas">
        <h2>Data Schemas</h2>
        {% for schema in schemas %}
        <div class="endpoint">
            <h3>{{ schema.name }}</h3>
            <p>{{ schema.description }}</p>
            {% if schema.properties %}
            <h4>Properties</h4>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th style="padding: 8px;">Name</th>
                    <th style="padding: 8px;">Type</th>
                    <th style="padding: 8px;">Required</th>
                    <th style="padding: 8px;">Description</th>
                </tr>
                {% for name, prop in schema.properties.items() %}
                <tr>
                    <td style="padding: 8px;">{{ name }}</td>
                    <td style="padding: 8px;">{{ prop.type }}</td>
                    <td style="padding: 8px;">{{ 'Yes' if name in schema.required else 'No' }}</td>
                    <td style="padding: 8px;">{{ prop.get('description', 'No description') }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </div>
        {% endfor %}
    </section>
    
    <section id="examples">
        <h2>Code Examples</h2>
        
        <h3>Python Example</h3>
        <div class="code">
            <pre>
import requests
import json

# Authentication
auth_response = requests.post('http://localhost:8000/auth/login', 
    json={'username': 'your_username', 'password': 'your_password'})
token = auth_response.json()['access_token']

# API request with authentication
headers = {'Authorization': f'Bearer {token}'}
response = requests.get('http://localhost:8000/api/health', headers=headers)
print(response.json())
            </pre>
        </div>
        
        <h3>JavaScript Example</h3>
        <div class="code">
            <pre>
// Authentication
const authResponse = await fetch('http://localhost:8000/auth/login', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({username: 'your_username', password: 'your_password'})
});
const {access_token} = await authResponse.json();

// API request with authentication
const response = await fetch('http://localhost:8000/api/health', {
    headers: {'Authorization': `Bearer ${access_token}`}
});
const data = await response.json();
console.log(data);
            </pre>
        </div>
        
        <h3>cURL Example</h3>
        <div class="code">
            <pre>
# Authentication
curl -X POST "http://localhost:8000/auth/login" \\
     -H "Content-Type: application/json" \\
     -d '{"username":"your_username","password":"your_password"}'

# API request (replace TOKEN with actual token)
curl -X GET "http://localhost:8000/api/health" \\
     -H "Authorization: Bearer TOKEN"
            </pre>
        </div>
    </section>
</body>
</html>"""
        
        with open(self.templates_dir / "api_docs.html", 'w') as f:
            f.write(html_template)
    
    def generate_documentation(
        self, 
        endpoints: List[APIEndpoint],
        schemas: List[APISchema],
        doc_type: DocumentationType = DocumentationType.API_REFERENCE,
        output_format: OutputFormat = OutputFormat.HTML,
        title: str = "Holistic Diagnostic Platform API",
        description: str = "Comprehensive medical imaging and AI diagnostic API",
        version: str = "1.0.0"
    ) -> str:
        """Generate comprehensive API documentation."""
        
        if output_format == OutputFormat.HTML:
            return self._generate_html_docs(endpoints, schemas, title, description, version)
        elif output_format == OutputFormat.MARKDOWN:
            return self._generate_markdown_docs(endpoints, schemas, title, description, version)
        elif output_format == OutputFormat.JSON:
            return self._generate_openapi_json(endpoints, schemas, title, description, version)
        elif output_format == OutputFormat.YAML:
            return self._generate_openapi_yaml(endpoints, schemas, title, description, version)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_html_docs(
        self,
        endpoints: List[APIEndpoint],
        schemas: List[APISchema],
        title: str,
        description: str,
        version: str
    ) -> str:
        """Generate HTML documentation."""
        
        env = Environment(loader=FileSystemLoader(str(self.templates_dir)))
        template = env.get_template('api_docs.html')
        
        html_content = template.render(
            title=title,
            description=description,
            version=version,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            endpoints=endpoints,
            schemas=schemas
        )
        
        output_file = self.output_dir / "api_documentation.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML documentation generated: {output_file}")
        return str(output_file)
    
    def _generate_markdown_docs(
        self,
        endpoints: List[APIEndpoint],
        schemas: List[APISchema],
        title: str,
        description: str,
        version: str
    ) -> str:
        """Generate Markdown documentation."""
        
        markdown_content = f"""# {title}

{description}

**Version:** {version}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Data Schemas](#data-schemas)
5. [Code Examples](#code-examples)

## Overview

This is the comprehensive API documentation for the Holistic Diagnostic Platform. The API provides endpoints for medical imaging analysis, AI-powered diagnostics, and clinical data management.

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## API Endpoints

"""
        
        for endpoint in endpoints:
            markdown_content += f"""
### {endpoint.method.upper()} {endpoint.path}

**Summary:** {endpoint.summary}

{endpoint.description}

"""
            
            if endpoint.parameters:
                markdown_content += "**Parameters:**\n\n"
                markdown_content += "| Name | Type | Required | Description |\n"
                markdown_content += "|------|------|----------|-------------|\n"
                
                for param in endpoint.parameters:
                    required = "Yes" if param.get('required', False) else "No"
                    description = param.get('description', 'No description')
                    markdown_content += f"| {param['name']} | {param['schema']['type']} | {required} | {description} |\n"
                
                markdown_content += "\n"
            
            if endpoint.responses:
                markdown_content += "**Responses:**\n\n"
                for status, response in endpoint.responses.items():
                    markdown_content += f"- **{status}:** {response['description']}\n"
                markdown_content += "\n"
        
        markdown_content += """
## Data Schemas

"""
        
        for schema in schemas:
            markdown_content += f"""
### {schema.name}

{schema.description}

"""
            if schema.properties:
                markdown_content += "**Properties:**\n\n"
                markdown_content += "| Name | Type | Required | Description |\n"
                markdown_content += "|------|------|----------|-------------|\n"
                
                for name, prop in schema.properties.items():
                    required = "Yes" if name in schema.required else "No"
                    description = prop.get('description', 'No description')
                    markdown_content += f"| {name} | {prop['type']} | {required} | {description} |\n"
                
                markdown_content += "\n"
        
        markdown_content += """
## Code Examples

### Python Example

```python
import requests
import json

# Authentication
auth_response = requests.post('http://localhost:8000/auth/login', 
    json={'username': 'your_username', 'password': 'your_password'})
token = auth_response.json()['access_token']

# API request with authentication
headers = {'Authorization': f'Bearer {token}'}
response = requests.get('http://localhost:8000/api/health', headers=headers)
print(response.json())
```

### JavaScript Example

```javascript
// Authentication
const authResponse = await fetch('http://localhost:8000/auth/login', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({username: 'your_username', password: 'your_password'})
});
const {access_token} = await authResponse.json();

// API request with authentication
const response = await fetch('http://localhost:8000/api/health', {
    headers: {'Authorization': `Bearer ${access_token}`}
});
const data = await response.json();
console.log(data);
```

### cURL Example

```bash
# Authentication
curl -X POST "http://localhost:8000/auth/login" \\
     -H "Content-Type: application/json" \\
     -d '{"username":"your_username","password":"your_password"}'

# API request (replace TOKEN with actual token)
curl -X GET "http://localhost:8000/api/health" \\
     -H "Authorization: Bearer TOKEN"
```
"""
        
        output_file = self.output_dir / "api_documentation.md"
        with open(output_file, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown documentation generated: {output_file}")
        return str(output_file)
    
    def _generate_openapi_json(
        self,
        endpoints: List[APIEndpoint],
        schemas: List[APISchema],
        title: str,
        description: str,
        version: str
    ) -> str:
        """Generate OpenAPI JSON specification."""
        
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": title,
                "description": description,
                "version": version,
                "contact": {
                    "name": "API Support",
                    "email": "support@example.com"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.example.com",
                    "description": "Production server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                }
            }
        }
        
        # Add endpoints
        for endpoint in endpoints:
            path = endpoint.path
            if path not in openapi_spec["paths"]:
                openapi_spec["paths"][path] = {}
            
            method = endpoint.method.lower()
            openapi_spec["paths"][path][method] = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "parameters": endpoint.parameters,
                "responses": endpoint.responses,
                "tags": endpoint.tags,
                "security": [{"bearerAuth": []}] if endpoint.security else []
            }
            
            if endpoint.request_body:
                openapi_spec["paths"][path][method]["requestBody"] = endpoint.request_body
        
        # Add schemas
        for schema in schemas:
            openapi_spec["components"]["schemas"][schema.name] = {
                "type": schema.type,
                "description": schema.description,
                "properties": schema.properties,
                "required": schema.required
            }
            
            if schema.example:
                openapi_spec["components"]["schemas"][schema.name]["example"] = schema.example
        
        output_file = self.output_dir / "openapi_spec.json"
        with open(output_file, 'w') as f:
            json.dump(openapi_spec, f, indent=2)
        
        logger.info(f"OpenAPI JSON specification generated: {output_file}")
        return str(output_file)
    
    def _generate_openapi_yaml(
        self,
        endpoints: List[APIEndpoint],
        schemas: List[APISchema],
        title: str,
        description: str,
        version: str
    ) -> str:
        """Generate OpenAPI YAML specification."""
        
        # Generate JSON first, then convert to YAML
        json_file = self._generate_openapi_json(endpoints, schemas, title, description, version)
        
        with open(json_file, 'r') as f:
            spec_data = json.load(f)
        
        output_file = self.output_dir / "openapi_spec.yaml"
        with open(output_file, 'w') as f:
            yaml.dump(spec_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"OpenAPI YAML specification generated: {output_file}")
        return str(output_file)

class DocumentationManager:
    """Main documentation management system."""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = source_dir
        self.analyzer = APIAnalyzer(source_dir)
        self.generator = DocumentationGenerator(output_dir)
    
    def generate_complete_documentation(self) -> Dict[str, str]:
        """Generate complete documentation suite."""
        
        # Analyze source code for endpoints and schemas
        app_files = list(Path(self.source_dir).glob("**/main.py"))
        app_files.extend(Path(self.source_dir).glob("**/app.py"))
        
        all_endpoints = []
        for app_file in app_files:
            endpoints = self.analyzer.analyze_fastapi_app(str(app_file))
            all_endpoints.extend(endpoints)
        
        # Extract schemas
        models_dirs = list(Path(self.source_dir).glob("**/models"))
        all_schemas = []
        for models_dir in models_dirs:
            schemas = self.analyzer.extract_schemas_from_models(str(models_dir))
            all_schemas.extend(schemas)
        
        # Generate documentation in all formats
        generated_files = {}
        
        for output_format in OutputFormat:
            try:
                file_path = self.generator.generate_documentation(
                    endpoints=all_endpoints,
                    schemas=all_schemas,
                    output_format=output_format
                )
                generated_files[output_format.value] = file_path
            except Exception as e:
                logger.error(f"Error generating {output_format.value} documentation: {str(e)}")
        
        return generated_files

# CLI Interface
def main():
    """Main CLI interface for documentation generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Documentation Generator")
    parser.add_argument("source_dir", help="Source code directory to analyze")
    parser.add_argument("--output-dir", "-o", default="./docs", help="Output directory for documentation")
    parser.add_argument("--format", choices=[f.value for f in OutputFormat], default="html", help="Output format")
    parser.add_argument("--title", default="API Documentation", help="Documentation title")
    parser.add_argument("--description", default="Generated API documentation", help="Documentation description")
    parser.add_argument("--version", default="1.0.0", help="API version")
    
    args = parser.parse_args()
    
    try:
        manager = DocumentationManager(args.source_dir, args.output_dir)
        
        if args.format == "all":
            # Generate all formats
            generated_files = manager.generate_complete_documentation()
            print("Generated documentation files:")
            for format_type, file_path in generated_files.items():
                print(f"  {format_type}: {file_path}")
        else:
            # Generate specific format
            analyzer = APIAnalyzer(args.source_dir)
            generator = DocumentationGenerator(args.output_dir)
            
            # Find FastAPI app files
            app_files = list(Path(args.source_dir).glob("**/main.py"))
            app_files.extend(Path(args.source_dir).glob("**/app.py"))
            
            all_endpoints = []
            for app_file in app_files:
                endpoints = analyzer.analyze_fastapi_app(str(app_file))
                all_endpoints.extend(endpoints)
            
            # Generate documentation
            output_format = OutputFormat(args.format)
            file_path = generator.generate_documentation(
                endpoints=all_endpoints,
                schemas=[],
                output_format=output_format,
                title=args.title,
                description=args.description,
                version=args.version
            )
            
            print(f"Documentation generated: {file_path}")
            
    except Exception as e:
        logger.error(f"Error generating documentation: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())