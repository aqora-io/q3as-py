# Getting Started

Q3AS stands for Quantum Algorithms as a Service, a hosted platform for you to run and iterate on quantum and hybrid quantum solutions.

## Installation

You can install the SDK from [PyPI](https://pypi.org/project/q3as) using [`pip`](https://pypi.org/project/pip/)

```bash
pip install q3as
```

## Getting an API Key

To run your algorithms in the cloud, you have to create an API key and load it into your `Credentials`

Start by visiting [https://q3as.aqora.io](https://q3as.aqora.io) and signing in with your GitHub or Google account. Click on your profile in the top right and go to **API Keys**. Tap **Add API Key** and enter a description for your API key. Tap **Copy JSON to clipboard** and paste the result in a file on your computer.

## Basic Usage

```python
from q3as import Client, Credentials, VQE
from q3as.app import Maxcut

credentials = Credentials.load("path/to/credentials.json")# (1)!
client = Client(credentials)

graph = [# (2)!
    (0, 1, 1.0),
    (0, 2, 1.0),
    (0, 4, 1.0),
    (1, 2, 1.0),
    (2, 3, 1.0),
    (3, 4, 1.0),
]

app = Maxcut(graph)# (3)!

job = VQE.builder().app(app).send(client)# (4)!

print(job.name)# (5)!

print(job.result())# (6)!
```

1. This path should match the path where you saved your API key JSON file.
2. We define a graph as a list of edges and their weights
3. Maximum cut is an NP hard problem used in optimization algorithms. You can learn more at [https://en.wikipedia.org/wiki/Maximum_cut](https://en.wikipedia.org/wiki/Maximum_cut)
4. We will use a Variational Quantum Eigensolver to solve the maximum cut problem we have defined, and send it to the Q3AS to run. You can learn more about VQE at [https://en.wikipedia.org/wiki/Variational_quantum_eigensolver](https://en.wikipedia.org/wiki/Variational_quantum_eigensolver)
5. Get the name of the Job. This can be used to find it on the platform and retrieve the Job results later
6. Get the result of the Job. This will block until the Job is finished.

Visit [https://q3as.aqora.io](https://q3as.aqora.io) to see your Job running in the cloud, and visualize the results
