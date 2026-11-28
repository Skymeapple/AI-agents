"""nexus: the connections between the projects in this repository.

sciscout finds and prices research directions, autoscout reads the literature,
aisci-verifiers checks claims, worldsim models global problems, and the
research catalogues list what is unsolved. Each was built on its own. The
modules here wire them to each other:

``loop``        verify claims per direction and let the results steer what is read next
``matching``    open problems against discovered directions: who is working on what
``impact``      a direction's success priced by what it does to world outcomes
``ledger``      every dated forecast the projects made, scored as reality arrives
``bottleneck``  physical scale-up limits in place of assumed commercialisation priors

They share sciscout's rule: every number says where it came from.
"""
