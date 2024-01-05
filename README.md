# MeshOps &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/readyplayerme/meshops/blob/main/LICENSE.txt)

A collection of tools for dealing with mesh related data.

## Setup

You'll find setup instruction of this project in the [CONTRIBUTING.md](https://github.com/readyplayerme/meshops/blob/main/CONTRIBUTING.md) file.

## UV Seams Transitioning

UV seams are splits in a triangle mesh that, however, is supposed to represent a continuous surface across these splits.
These splits are necessary to allow the mesh to flatten out cleanly in UV space and have as little distortion in the texture projection as possible.
The UV seams are splits between UV islands for which the projection onto the mesh should appear as seamless as possible.

### Goal

- identify UV splits in a mesh
- mitigate the mismatch between image content when transitioning over a UV split
