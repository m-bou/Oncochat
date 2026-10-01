## Objective and success criteria

This take-home exercise is designed to evaluate your ability to quickly design and develop a first working version of an agentic product.

Please note the following expectations:

- The solution does not need to be perfect or fully optimized.

- The priority is to deliver a functional solution that we can run seamlessly on our local machines.

- Meeting every single requirement is less important than providing a runnable, consistent implementation.

- We expect you to spend no more than 4 hours on this exercise.

- You are allowed to use coding assistants.

- If successful, you will be invited to present your solution to a panel, where we will discuss your design decisions and trade-offs.

time constraints.

The goal is to assess your pragmatism, technical judgment, and ability to deliver under

## Expected behavior

You are expected to build the first proof of concept of an agentic product allowing non-technical stakeholders to interact in natural language. The solution should be able to leverage and orchestrate two existing functionalities that are able to perform basic analyses on a dataset (see section Available data and functions below). The solution is expected to provide relevant answers to the following queries:

- How can you help me?

- What are the main genes involved in lung cancer?

- What is the median value expression of genes involved in breast cancer?

- What is the median value expression of genes involved in esophageal cancer?

## Technical requirements

- The prototype is provided as a github repository and can be run without you on another laptop

- README markdown file with instructions on how to run and an overview of the design

- A web application running in Docker is preferred but a command line application running in a terminal is also acceptable

- Language: Python

- Runtime environment - the prototype must execute on a standard Mac (Apple Silicon or Intel) and Windows 11 laptop with <= 16 GB RAM.

- Don’t depend on a specific GPU

- The README or a similar file describes the architecture, use and tradeoffs of AI components in the prototype

- The README or a similar file describes the pros and cons of using AI-assisted coding in this case


## Available data and functions

- A csv is attached with this document (wo_data.csv)

- You can rely on the following script gathering two functions.

```
Python
import pandas as pd
import random
from typing import List, Dict
# Load CSV
df = pd.read_csv("wo_data.csv")
def get_targets(cancer_name: str) -> List[str]:
"""Return a list of genes for a given cancer type."""
return df[df['cancer_indication'] == cancer_name]['gene'].tolist()
def get_expressions(genes: List[str]) -> Dict[str, float]:
"""Return the median values for the given list of genes."""
subset = df[df['gene'].isin(genes)]
return dict(zip(subset['gene'], subset['median_value']))
# Example usage
if __name__ == "__main__":
cancer = "lung"
targets = get_targets(cancer)
expressions = get_expressions(targets)
print(f"{cancer} targets:", targets)
print(f"Expression values:", expressions)
```
