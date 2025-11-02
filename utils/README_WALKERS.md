# Image Walker Library

A comprehensive library for visualizing and experimenting with different image traversal strategies.

## Overview

This library explores the question: **"How do we order pixels/regions in an image for sequential prediction?"**

It implements various content-aware walk strategies that traverse images following gradients, saliency, edges, or superpixel structures.

## Quick Start

```python
from utils.image_walker import ImageWalker, BrightnessGradientWalk
import cv2

# Load image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create walker
walker = ImageWalker(image)

# Execute a walk strategy
strategy = BrightnessGradientWalk(maximize=True)
path = walker.walk(strategy, max_steps=1000)

# Visualize
viz = walker.visualize(path)
cv2.imwrite('walk_visualization.png', cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
```

## Implemented Walk Strategies

### Pixel-Level Walks

1. **BrightnessGradientWalk** - Follow direction of greatest/least sum(R+G+B) change
   ```python
   BrightnessGradientWalk(maximize=True)   # Follow edges/boundaries
   BrightnessGradientWalk(maximize=False)  # Follow smooth regions
   ```

2. **StochasticGradientWalk** - Like SGD, gradient walk with randomization
   ```python
   StochasticGradientWalk(temperature=1.0, maximize=True)
   ```

3. **ColorChannelGradientWalk** - Follow gradient in specific RGB channel
   ```python
   ColorChannelGradientWalk(channel=0, maximize=True)  # Red channel
   ```

4. **NoBacktrackingMinChangeWalk** 🔥 - Greedy walk in direction of smallest change
   ```python
   NoBacktrackingMinChangeWalk()  # Never retraces steps
   ```

5. **SaliencyWalk** - Follow high-contrast, visually salient regions
   ```python
   SaliencyWalk()  # Mimics human attention
   ```

6. **CenterOutwardWalk** - Distance-based from center
   ```python
   CenterOutwardWalk(reverse=False)  # Center → edges
   CenterOutwardWalk(reverse=True)   # Edges → center
   ```

7. **SpiralWalk** - Spiral pattern from center
   ```python
   SpiralWalk()
   ```

8. **EdgeFollowingWalk** - Follow edges detected by Canny
   ```python
   EdgeFollowingWalk(edge_strength_threshold=0.5)
   ```

9. **RandomWalk** - Baseline comparison
   ```python
   RandomWalk()
   ```

### Superpixel-Level Walks

```python
from utils.superpixel_walker import SuperpixelWalker

# Create superpixel walker
sp_walker = SuperpixelWalker(image, method='slic', n_segments=100)

# Different ordering strategies
order = sp_walker.walk_by_size(largest_first=True)
order = sp_walker.walk_by_brightness(brightest_first=True)
order = sp_walker.walk_by_position(start_corner='center')
order = sp_walker.walk_by_gradient(maximize=True)
order = sp_walker.walk_by_color_variance(highest_first=True)
order = sp_walker.walk_adjacency_graph(strategy='bfs')

# Visualize
viz = sp_walker.visualize_superpixels(order, show_order=True)
```

## Superpixel Algorithms

Implemented:
- **SLIC** - Fast, well-balanced (default)
- **Felzenszwalb** - Excellent boundary adherence
- **QuickShift** - Good texture preservation
- **Watershed** - Separates touching objects

See `superpixel_walker.py` header for complete list of 10 algorithms with comparison.

## Creating Custom Walk Strategies

```python
from utils.image_walker import WalkStrategy
import numpy as np

class MyCustomWalk(WalkStrategy):
    def compute_score(self, image, position, visited, walk_history):
        row, col = position
        # Your scoring logic here
        # Higher score = more likely to visit next
        return your_score

    def get_name(self):
        return "My Custom Walk"

# Use it
walker = ImageWalker(image)
path = walker.walk(MyCustomWalk())
```

## Run Demos

```bash
# Create synthetic test image and run all demos
python experiments/demo_image_walks.py

# Outputs saved to experiments/outputs/
# - pixel_walks_comparison.png (10 strategies compared)
# - superpixel_walks_comparison.png (4 methods × 4 orderings)
# - Walk statistics printed to console
```

## Use Cases

### 1. Sequential Token Generation
Create content-aware orderings for autoregressive image models:

```python
# Walk determines token sequence
strategy = BrightnessGradientWalk(maximize=True)
path = walker.walk(strategy)

# Use path ordering for training data
tokens = [image[step.position] for step in path]
```

### 2. Saliency-Based Attention
Mimic human visual attention patterns:

```python
strategy = SaliencyWalk()
path = walker.walk(strategy, max_steps=100)  # First 100 fixations
```

### 3. Edge-First Processing
Process edges/boundaries before smooth regions:

```python
strategy = EdgeFollowingWalk()
path = walker.walk(strategy)
```

### 4. Superpixel Ordering by Size
Order from largest to smallest semantic regions:

```python
sp_walker = SuperpixelWalker(image)
order = sp_walker.walk_by_size(largest_first=True)

# Do something with each superpixel in order
for sp_id in order:
    superpixel = sp_walker.superpixels[sp_id]
    # Process superpixel
    print(f"SP {sp_id}: {superpixel.area} pixels at {superpixel.center}")
```

## API Reference

### ImageWalker

```python
walker = ImageWalker(image: np.ndarray)

path = walker.walk(
    strategy: WalkStrategy,
    start_position: Optional[Tuple[int, int]] = None,  # Default: center
    max_steps: Optional[int] = None,  # Default: all pixels
    connectivity: int = 8  # 4 or 8 connected neighbors
) -> List[WalkStep]

viz = walker.visualize(
    walk_path: List[WalkStep],
    line_thickness: int = 1,
    color: Tuple[int, int, int] = (255, 0, 0),
    show_start: bool = True,
    show_end: bool = True
) -> np.ndarray
```

### SuperpixelWalker

```python
sp_walker = SuperpixelWalker(
    image: np.ndarray,
    method: str = 'slic',  # 'slic', 'felzenszwalb', 'quickshift', 'watershed'
    n_segments: int = 100,
    compactness: float = 10.0  # SLIC only
)

# Ordering methods
order = sp_walker.walk_by_size(largest_first: bool = True)
order = sp_walker.walk_by_brightness(brightest_first: bool = True)
order = sp_walker.walk_by_position(start_corner: str = 'center')
order = sp_walker.walk_by_gradient(maximize: bool = True)
order = sp_walker.walk_by_color_variance(highest_first: bool = True)
order = sp_walker.walk_adjacency_graph(start_id: Optional[int] = None,
                                       strategy: str = 'bfs')

viz = sp_walker.visualize_superpixels(order: Optional[List[int]] = None,
                                      show_order: bool = True)
```

## Dependencies

```
numpy
opencv-python (cv2)
scipy
scikit-image
matplotlib (for demos)
networkx
```

## Performance Notes

- **Pixel-level walks**: Fast for limited steps (~1000). Slow for full image traversal.
- **Superpixel walks**: Much faster, processes ~50-200 regions instead of thousands of pixels.
- **Gradient computation**: Cached on first call, reused for all positions.
- **Visualization**: Fast, uses OpenCV for drawing.

## Future Ideas

- [ ] Learned walk policies (RL)
- [ ] Multi-scale hierarchical walks
- [ ] Temporal walks for videos
- [ ] Parallel/distributed walks
- [ ] GPU acceleration for large images
- [ ] More superpixel algorithms (SEEDS, LSC, ERS)

## References

- Gradient-based walks inspired by visual attention and saccadic eye movements
- Superpixel methods from scikit-image documentation
- Part of the image-ssl repository exploring SSL techniques
