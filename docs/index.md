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

Once you've created an API Key and installed the SDK you're ready to create your first quantum job! We're going to first create our client so we can jobs to Q3AS. To do that just load the credentials you've saved in the previous step.

```python
from q3as import Client, Credentials

credentials = Credentials.load("path/to/credentials.json")# (1)!
client = Client(credentials)
```

1. This path should match the path where you saved your API key JSON file.

From here we can start building out the definition of the problem that we would like to solve. We'll start with an NP Hard problem called [Maximum Weighted Cut](https://en.wikipedia.org/wiki/Maximum_cut). We can define a graph we would like to cut by supplying a list of edges and their weights.

```python
graph = [
    (0, 1, 1.0),
    (0, 2, 1.0),
    (0, 4, 1.0),
    (1, 2, 1.0),
    (2, 3, 1.0),
    (3, 4, 1.0),
]
```

We can then give this graph to our "Application" which will define what we want to do with it and how to translate it into the quantum world and back. Q3AS defines multiple such problem domains that you can use.

```python
from q3as.app import Maxcut

app = Maxcut(graph)
```

We now need to define a solver for our problem. We will use a [Variational Quantum Eigensolver](https://en.wikipedia.org/wiki/Variational_quantum_eigensolver) or VQE for short

```python
from q3as import VQE

vqe = VQE.builder().app(app)
```

Now we can send the job to the Q3AS, and let the server handle the computation and the visualization of the intermediate results

```python
job = vqe.send(client)
# get the name of the job
print(job.name)
# wait for and retrieve the results of the job
print(job.result())
```

Putting it all together we have!

```python
from q3as import Client, Credentials, VQE
from q3as.app import Maxcut

client = Client(Credentials.load("credentials.json"))

job = (
    VQE
        .builder()
        .app(
            Maxcut([
                (0, 1, 1.0),
                (0, 2, 1.0),
                (0, 4, 1.0),
                (1, 2, 1.0),
                (2, 3, 1.0),
                (3, 4, 1.0),
            ])
        )
        .send(client)
)
print(job.name)
print(job.result())
```
