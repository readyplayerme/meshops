# Texture Synthesis

A collection of tools for texture mapping and texture synthesis.

## UV Seams Matcher

UV seams are splits in a triangle mesh that, however, is supposed to represent a continuous surface across these splits.
These splits are necessary to allow the mesh to flatten out cleanly in UV space and have as little distortion in the texture projection as possible.
The UV seams are splits between UV islands for which the projection onto the mesh should appear as seamless as possible.

### Goal

- identify UV borders in a mesh
- pair UV borders that are likely to be seams between UV islands
- mitigate the mismatch between image content from one border to the other
