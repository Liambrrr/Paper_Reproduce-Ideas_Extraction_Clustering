import os
import boto3
from botocore.exceptions import ClientError
from openai import OpenAI, AzureOpenAI
import json
import regex
import json
import time
import pandas as pd
from anytree import Node
import traceback
import subprocess
import tiktoken
from sklearn import metrics
import numpy as np

class APIClient:
    """
    Prompting for OpenAI, VertexAI, vLLM, Gemini, Azure, Ollama, and OpenRouter.

    Parameters:
    - api: API type (e.g., 'openai', 'vertex', 'vllm', 'ollama', 'gemini', 'azure', 'openrouter')
    - model: Model name

    Methods:
    - estimate_token_count: Estimate token count for a prompt
    - truncating: Truncate document to max tokens
    - iterative_prompt: Prompt API one by one with retries
    - batch_prompt: Batch prompting for vLLM API
    """

    def __init__(self, api, model, host=None):
        self.api = api
        self.model = model
        self.client = None

        # Setting API key ----
        if api == "openai":
            # Standard OpenAI client, expects OPENAI_API_KEY
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        elif api == "openrouter":
            # OpenRouter is OpenAI-compatible but uses a different base URL and key
            # Required env var: OPENROUTER_API_KEY
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.client = OpenAI(
                base_url=base_url,
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        elif api == "bedrock":
            # AWS Bedrock Runtime client, using your AWS credentials / env / ~/.aws/credentials
            region = os.environ.get("BEDROCK_REGION", "us-west-2")
            self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)
            # For Bedrock, `model` is the Bedrock model ID, e.g. "meta.llama3-1-405b-instruct-v1:0"
            self.model_id = model

        elif api == "vertex":
            vertexai.init(
                project=os.environ["VERTEX_PROJECT"],
                location=os.environ["VERTEX_LOCATION"],
            )
            if model.startswith("gemini"):
                self.model_obj = genai.GenerativeModel(self.model)

        elif api == "ollama":
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            )

        elif api == "vllm":
            self.hf_token = os.environ.get("HF_TOKEN")
            self.llm = LLM(
                self.model,
                download_dir=os.environ.get("HF_HOME", None),
            )
            self.tokenizer = self.llm.get_tokenizer()

        elif api == "gemini":
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            self.model_obj = genai.GenerativeModel(self.model)

        elif api == "azure":
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01",
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            )

        else:
            raise ValueError(
                f"API {api} not supported. Custom implementation required."
            )

    def estimate_token_count(self, prompt: str) -> int:
        """
        Estimating the token count for the prompt with tiktoken
        """
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(prompt))

    def truncating(self, document: str, max_tokens: int) -> str:
        """
        Truncate the document to at most max_tokens tokens.
        """
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            print("Warning: model not found. Using o200k_base encoding.")
            enc = tiktoken.get_encoding("o200k_base")

        tokens = enc.encode(document)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return enc.decode(tokens)

    def iterative_prompt(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        system_message: str = "You are a helpful assistant.",
        num_try: int = 3,
        verbose: bool = False,
    ):
        """
        Prompting API one by one with retries
        """
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(num_try):
            try:
                if self.api in ["openai", "azure", "ollama", "openrouter"]:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=message,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    if verbose and hasattr(completion, "usage") and completion.usage:
                        print(
                            "Prompt token usage:",
                            completion.usage.prompt_tokens,
                            f"~${completion.usage.prompt_tokens/1000000*5}",
                        )
                        print(
                            "Response token usage:",
                            completion.usage.completion_tokens,
                            f"~${completion.usage.completion_tokens/1000000*15}",
                        )
                    return completion.choices[0].message.content
                
                elif self.api == "bedrock":
                    # Use AWS Bedrock "converse" API with Llama 3.1 Instruct model
                    user_text = f"{system_message}\n\n{prompt}"

                    conversation = [
                        {
                            "role": "user",
                            "content": [{"text": user_text}],
                        }
                    ]

                    try:
                        response = self.bedrock_client.converse(
                            modelId=self.model_id,
                            messages=conversation,
                            inferenceConfig={
                                "maxTokens": max_tokens,
                                "temperature": temperature,
                                "topP": top_p,
                            },
                        )
                        # Extract the generated text
                        return response["output"]["message"]["content"][0]["text"]
                    except ClientError as e:
                        print(f"Bedrock ClientError: {e}")
                        time.sleep(60)
                    except Exception as e:
                        print(f"Bedrock error: {e}")
                        time.sleep(60)

                elif self.api == "vertex":
                    if self.model.startswith("claude"):
                        client = AnthropicVertex(
                            region=os.environ["VERTEX_LOCATION"],
                            project_id=os.environ["VERTEX_PROJECT"],
                        )
                        message_obj = client.messages.create(
                            model=self.model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system=system_message,
                            messages=[message[1]],
                        )
                        message_json_str = message_obj.model_dump_json(indent=2)
                        message_dict = json.loads(message_json_str)
                        text_content = message_dict["content"][0]["text"]
                        if verbose:
                            print(
                                "Prompt usage:",
                                message_dict["usage"]["input_tokens"],
                                f"${message_dict['usage']['input_tokens']/1000000*3}",
                            )
                            print(
                                "Response usage:",
                                message_dict["usage"]["output_tokens"],
                                f"${message_dict['usage']['output_tokens']/1000000*15}",
                            )
                        return text_content
                    else:
                        config = GenerationConfig(
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        safety_config = [
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=HarmBlockThreshold.BLOCK_NONE,
                            ),
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=HarmBlockThreshold.BLOCK_NONE,
                            ),
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=HarmBlockThreshold.BLOCK_NONE,
                            ),
                            SafetySetting(
                                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=HarmBlockThreshold.BLOCK_NONE,
                            ),
                        ]
                        try:
                            response = self.model_obj.generate_content(
                                system_message + prompt,
                                generation_config=config,
                                safety_settings=safety_config,
                            )
                            return response.text.strip()
                        except:
                            traceback.print_exc()
                            time.sleep(60)

                elif self.api == "vllm":
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stop_token_ids=[
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                        ],
                    )
                    final_prompt = self.tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    vllm_output = self.llm.generate([final_prompt], sampling_params)
                    return [output.outputs[0].text for output in vllm_output][0]

                elif self.api == "gemini":
                    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                    self.model_obj = genai.GenerativeModel(self.model)
                    config = genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    safety_config = {
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                    try:
                        response = self.model_obj.generate_content(
                            system_message + prompt,
                            generation_config=config,
                            safety_settings=safety_config,
                        )
                        return response.text.strip()
                    except:
                        traceback.print_exc()
                        time.sleep(60)

            except Exception as e:
                print(f"Attempt {attempt + 1}/{num_try} failed: {e}")
                if attempt < num_try - 1:
                    time.sleep(60)
                else:
                    raise

    def batch_prompt(
        self,
        prompts: list,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        system_message: str = "You are a helpful assistant.",
    ):
        """
        Batch prompting for vLLM API
        """
        if self.api != "vllm":
            raise ValueError("Batch prompting not supported for this API.")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        )

        prompt_formatted = [
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        final_prompts = [
            self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in prompt_formatted
        ]
        outputs = self.llm.generate(final_prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
class TopicTree:
    """
    Represents a hierarchical structure of topics.

    Parameters:
    - root_name: Name of the root topic

    Attributes:
    - root: Root node of the tree
    - level_nodes: Dictionary of nodes by level

    Methods:
    - node_to_str: Convert a node to a string representation
    - from_topic_list: Construct a TopicTree from a list of topic strings
    - from_seed_file: Construct a TopicTree from a seed file
    - _add_node: Add a node to the tree
    - _remove_node_by_name_lvl: Remove a node by name and level
    - to_prompt_view: Generate a string representation of the tree with indentation by level
    - find_duplicates: Find nodes with the same name and level in the tree
    - to_file: Save the tree to a file
    - to_topic_list: Convert the tree to a list of topic strings
    - get_root_descendants_name: Get the root description
    - update_tree: Update the topic tree by merging a set of topics into a new topic
    """

    def __init__(self, root_name="Topics"):
        self.root = Node(name=root_name, lvl=0, count=1, desc="Root topic", parent=None)
        self.level_nodes = {0: self.root}

    @staticmethod
    def node_to_str(node, count=True, desc=True):
        """
        Convert a node to a string representation.

        Parameters:
        - node: Node to convert
        - count: Include count in the string
        - desc: Include description in the string

        Returns:
        - str: String representation of the node
        """
        if not count and not desc:
            return f"[{node.lvl}] {node.name}"
        elif not count and desc:
            return f"[{node.lvl}] {node.name}: {node.desc}"
        elif count and not desc:
            return f"[{node.lvl}] {node.name} (Count: {node.count})"
        else:
            return f"[{node.lvl}] {node.name} (Count: {node.count}): {node.desc}"

    @staticmethod
    def from_topic_list(topic_src, from_file=False):
        """
        Construct a TopicTree from a list of topic strings or a file.

        Parameters:
        - topic_src: List of topic strings or path to a file
        - from_file: Flag to indicate if the source is a file

        Returns:
        - tree: Constructed TopicTree
        """
        tree = TopicTree()
        topic_list = open(topic_src, "r").readlines() if from_file else topic_src
        topic_list = [topic for topic in topic_list if len(topic.strip()) > 0]
        pattern = regex.compile(r"^\[(\d+)\] (.+) \(Count: (\d+)\)\s?:(.+)?")

        for topic in topic_list:
            if not topic.strip():
                continue
            try:
                match = regex.match(pattern, topic.strip())
                lvl, label, count, desc = (
                    int(match.group(1)),
                    match.group(2).strip(),
                    int(match.group(3)),
                    match.group(4).strip() if match.group(4) else "",
                )
            except:
                print(match)
                print("Error reading", topic)
                traceback.print_exc()

            tree._add_node(lvl, label, count, desc, tree.level_nodes.get(lvl - 1))

        return tree

    def from_seed_file(self, seed_file):
        """
        Construct a TopicTree from a seed file (no description/count)

        Parameters:
        - seed_file: Path to the seed file

        Returns:
        - tree: Constructed TopicTree
        """
        tree = TopicTree()
        topic_list = open(seed_file, "r").readlines() if seed_file else []
        topic_list = [topic for topic in topic_list if len(topic.strip()) > 0]
        pattern = regex.compile(r"^\[(\d+)\] (.+)")

        for topic in topic_list:
            if not topic.strip():
                continue
            try:
                match = regex.match(pattern, topic.strip())
                lvl, label = (
                    int(match.group(1)),
                    match.group(2).strip(),
                )
            except:
                print(match)
                print("Error reading", topic)
                traceback.print_exc()

            tree._add_node(lvl, label, 1, "", tree.level_nodes.get(lvl - 1))

        return tree

    def _add_node(self, lvl, label, count, desc, parent_node):
        """
        Add a node to the tree, merging with duplicates if present.

        Parameters:
        - lvl: Level of the node
        - label: Name of the node
        - count: Count of the node
        - desc: Description of the node
        - parent_node: Parent node of the new node
        """
        if parent_node:
            existing = next((n for n in parent_node.children if n.name == label), None)
            if existing:
                existing.count += count
            else:
                new_node = Node(
                    name=label, lvl=lvl, count=count, desc=desc, parent=parent_node
                )
                self.level_nodes[lvl] = new_node

    def _remove_node_by_name_lvl(self, name, lvl):
        """
        Remove a node by name and level.

        Parameters:
        - name: Name of the node
        - lvl: Level of the node
        """
        node = next(
            (n for n in self.root.descendants if n.name == name and n.lvl == lvl), None
        )
        if node:
            node.parent = None

    def to_prompt_view(self, desc=True):
        """
        Generate a string representation of the tree with indentation by level.

        Parameters:
        - desc: Include description in the string

        Returns:
        - str: String representation of the tree
        """

        def traverse(node, result=""):
            if node.lvl > 0:
                result += (
                    "\t" * (node.lvl - 1)
                    + self.node_to_str(node, count=False, desc=False)
                    + "\n"
                )
            for child in node.children:
                result = traverse(child, result)
            return result

        return traverse(self.root)

    def find_duplicates(self, name, level):
        """
        Find nodes with the same name and level in the tree.

        Parameters:
        - name: Name of the node
        - level: Level of the node

        Returns:
        - list: List of nodes with the same name and level
        """
        return [
            node
            for node in self.root.descendants
            if node.name.lower() == name.lower() and node.lvl == level
        ]

    def to_file(self, fname):
        """
        Save the tree to a file.

        Parameters:
        - fname: Path to the file
        """
        with open(fname, "w") as f:
            for node in self.root.descendants:
                if len(node.desc) > 0:
                    indentation = "    " * (node.lvl - 1)
                    f.write(indentation + self.node_to_str(node) + "\n")

    def to_topic_list(self, desc=True, count=True):
        """
        Convert the tree to a list of topic strings.

        Parameters:
        - desc: Include description in the string
        - count: Include count in the string

        Returns:
        - list: List of topic strings
        """
        return [self.node_to_str(node, count, desc) for node in self.root.descendants]

    def get_root_descendants_name(self):
        """Get the root description.

        Returns:
        - list: List of root descendants' names
        """
        return [node.name for node in self.root.descendants]

    def update_tree(self, original_topics, new_topic_name, new_topic_desc):
        """
        Update the topic tree by merging a set of topics into a new topic if
        the final count is <= the total count of all topics.

        Parameters:
        - original_topics: List of tuples (name, level) of topics to merge.
        - new_topic_name: Name of the new merged topic.
        - new_topic_desc: Description for the new merged topic.

        Returns:
        - tree: Updated TopicTree
        """
        total_count = 0
        parent_node = None
        nodes_to_merge = []

        for name, lvl in original_topics:
            duplicates = self.find_duplicates(name, lvl)
            nodes_to_merge.extend(duplicates)
            total_count += sum(node.count for node in duplicates)
            if duplicates and not parent_node:
                parent_node = duplicates[0].parent

        if parent_node is None:
            parent_node = self.root

        merged_topic_node = (
            next(
                (node for node in parent_node.children if node.name == new_topic_name),
                None,
            )
            if parent_node
            else None
        )

        if merged_topic_node:
            final_count = total_count
            if final_count <= total_count:
                merged_topic_node.count = final_count
                merged_topic_node.desc = new_topic_desc
        else:
            if total_count <= sum(node.count for node in nodes_to_merge):
                merged_topic_node = self._add_node(
                    lvl=parent_node.lvl + 1,
                    label=new_topic_name,
                    count=total_count,
                    desc=new_topic_desc,
                    parent_node=parent_node,
                )

        for node in nodes_to_merge:
            if node != merged_topic_node:
                self._remove_node_by_name_lvl(node.name, node.lvl)

        return self
