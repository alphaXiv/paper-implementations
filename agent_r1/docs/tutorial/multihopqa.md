### Tutorial: Customizing Tools for Multi-hop Question Answering Tasks

This tutorial demonstrates how to create and customize tools for Agent-based multi-hop question answering. Multi-hop QA is a challenging task that requires reasoning across multiple documents to derive answers, making it an excellent use case for showcasing tool customization.

The tutorial covers four main components:
1. Setting up the retrieval service for knowledge access
2. Creating custom tools for Agent interaction
3. Preparing datasets (HotpotQA, 2Wiki, and MusiQue)
4. Training your model using GRPO

#### 1. Setting up Retrieval Service

##### Download Index and Corpus
Download the required data from Hugging Face:
- Corpus: `corag/kilt-corpus`
- Index: `russwest404/kilt_index`

```bash
# Create directory for data
mkdir -p data/corpus/kilt

# Download corpus and index (using Hugging Face CLI or web download)
huggingface-cli download corag/kilt-corpus
huggingface-cli download russwest404/kilt_index --local-dir data/corpus/kilt
```

##### Configure and Start Retrieval Service
Modify the `INDEX_PATH` in `scripts/kilt_search_server/run_search_api.sh` to point to the correct index file location:

```bash
# Edit script to modify INDEX_PATH
export INDEX_PATH="../../data/corpus/kilt/kilt_index_IVF16384_PQ64.bin"
```

Run the script to start the retrieval service:

```bash
cd scripts/kilt_search_server
bash run_search_api.sh
```

##### Optional: Build Your Own Index
The default index uses inverted indexing and PQ quantization, which may reduce retrieval performance. If your hardware resources allow, consider building your own index.

We provide sample code in `scripts/kilt_search_server/process_kilt.py`, which you can configure for different index types:

```bash
python process_kilt.py --index_type "HNSW64"
```

Supported index types include:
- `Flat`: Exact but memory-intensive
- `IVF4096,Flat`: Balance between performance and accuracy
- `IVF4096,PQ96`: Compressed index, sacrificing some accuracy
- `HNSW32`/`HNSW64`: Efficient approximate nearest neighbor search

#### 2. Creating Custom Tools for Agent Interaction

Custom tools allow your Agent to interact with external services and perform specific actions. For the multi-hop QA task, we'll create a tool that can search for information across multiple knowledge sources.

##### Tool Structure

Each tool is built on the `BaseTool` class (located at `agent_r1/tool/base.py`), which defines a common interface:

```python
class BaseTool(ABC):
    name: str = ''              # Tool name used in function calling
    description: str = ''       # Description explaining what the tool does
    parameters: dict = {}       # JSON schema defining expected parameters
    
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        # Implementation of the tool logic
        pass
```

##### Creating a Wiki Search Tool

Let's examine how to implement a custom tool for searching Wikipedia. This implementation can be found at `agent_r1/tool/tools/wiki_search_tool.py`. This tool will be used by the Agent to retrieve information needed to answer multi-hop questions.

1. Define the tool class with its metadata:

```python
class WikiSearchTool(BaseTool):
    name = "search"
    description = "Search for information on the internet using Wikipedia as a knowledge source."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
```

2. Implement the `execute` method to handle search requests:

```python
def execute(self, args: Dict) -> Dict[str, Any]:
    """
    Execute search query
    
    Args:
        args: Tool parameters, containing:
            - "query": search query string
        
    Returns:
        Dict with format {"content": result_content, "success": bool}
    """
    query = args.get("query", "").strip()
    
    # Call the search API
    response = requests.get(
        f"{self.api_url}/search",
        params={"query": query, "top_k": limit}
    )
    
    # Process and format results
    if response.status_code == 200:
        result = response.json()
        formatted_result = self._format_results(result)
        return {"content": formatted_result, "success": True}
    else:
        return {"content": "Search failed", "success": False}
```

##### Integrating Custom Tools

To make your tools available to the Agent, you'll need to register them in your tool environment. This is typically done in the tool registry file `agent_r1/tool/tools/__init__.py`:

```python
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool

def _default_tool(name):
    # ...
    elif name == "wiki_search":
        return WikiSearchTool()
    # ...
```

This factory function approach allows tools to be created on demand by name. When setting up your environment, you can specify which tools to enable through configuration, and the environment will use this registry to create the appropriate tool instances.

##### Creating Your Own Tools

To create your own custom tool:

1. Subclass `BaseTool` and define the required attributes:
   - `name`: A unique identifier for the tool
   - `description`: Clear instructions on what the tool does and when to use it
   - `parameters`: A JSON schema defining the expected input parameters

2. Implement the `execute` method to handle the tool's logic and return results in a consistent format.

3. Optional: Implement `batch_execute` for optimized batch processing if your tool might handle multiple similar requests at once.

For more complex tools, consider how to properly handle errors, format responses for the Agent, and maintain stateful connections to external services if needed.

#### 3. Preparing Data

##### HotpotQA

```bash
# Create data directory
mkdir -p data/hotpotqa

# Run preprocessing script
python examples/data_preprocess/hotpotqa.py --local_dir ./data/hotpotqa
```

The preprocessing script will automatically download the HotpotQA dataset and convert it to the required format for training, saving it as `train.parquet` and `validation.parquet`.

##### 2Wiki

```bash
# Create data directory
mkdir -p data/2wiki

# Run preprocessing script
python examples/data_preprocess/2wikimultihopqa.py --local_dir ./data/2wiki
```

The preprocessing script will automatically download the 2WikiMultihopQA dataset and convert it to the required format, saving it as `train_processed.parquet` and `validation_processed.parquet`.

##### MusiQue

```bash
# Create data directory
mkdir -p data/musique

# Run preprocessing script, using answerable configuration by default
python examples/data_preprocess/musique.py --local_dir ./data/musique --config answerable
```

Preprocessing will generate `train_answerable_processed.parquet` and `validation_answerable_processed.parquet` files.

#### 4. Training the Model

##### Prepare Training Script
Copy the training script to the main directory:

```bash
cp examples/trainer/run_grpo_multihopqa.sh ./
```

##### Configure Training Parameters
Edit the `run_grpo_multihopqa.sh` script to adjust parameters according to your needs.

##### Run Training
Execute the training script:

```bash
bash run_grpo_multihopqa.sh
```

Training progress and logs will be saved to Wandb (if configured) and the console.
