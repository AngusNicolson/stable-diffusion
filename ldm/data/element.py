
import itertools

from elements.classes import ElementDataset


class Element(ElementDataset):
    def __init__(self, dataset_name, n, img_size=256, element_n=4, element_size=64, element_size_delta=16, element_seed=42, loc_seed=123):
        class_configs, allowed = select_dataset(dataset_name)
        super().__init__(
            allowed,
            class_configs,
            n,
            img_size,
            element_n,
            element_size,
            element_size_delta,
            element_seed,
            loc_seed
        )

    def __getitem__(self, idx):
        item = self.get_item(idx)
        img = self.transform(item.img)
        return {"image": img.permute(1, 2, 0), "class_label": item.class_labels_oh}


def select_dataset(name):
    if name == "simple":
        class_configs = [
            {"shape": None, "color": None, "texture": "solid"},
            {"shape": None, "color": "red", "texture": "solid"},
            {"shape": None, "color": "blue", "texture": "stripes_diagonal"},
            {"shape": None, "color": "green", "texture": "spots_polka"},
            {"shape": "circle", "color": None, "texture": "solid"},
            {"shape": "circle", "color": None, "texture": "spots_polka"},
            {"shape": "triangle", "color": "green", "texture": None},
            {"shape": "square", "color": "blue", "texture": None},
            {"shape": "triangle", "color": "red", "texture": "stripes_diagonal"},
            {"shape": "triangle", "color": "blue", "texture": "stripes_diagonal"},
            {"shape": "square", "color": "green", "texture": "spots_polka"},
            {"shape": "plus", "color": "magenta", "texture": "spots_polka"},
        ]

        allowed_shapes = ['square', 'circle', 'triangle', 'plus']
        allowed_colors = ['red', 'green', 'blue', 'magenta']
        allowed_textures = ["solid", "spots_polka", "stripes_diagonal"]
    elif name == "simple_all":
        allowed_shapes = ['square', 'circle', 'triangle', 'plus']
        allowed_colors = ['red', 'green', 'blue']
        allowed_textures = ["solid", "spots_polka", "stripes_diagonal"]

        allowed_config = [
            v + [None] for v in [allowed_shapes, allowed_colors, allowed_textures]
        ]
        class_configs = list(itertools.product(*allowed_config))

        class_configs = [
            v for v in class_configs if sum([in_v is None for in_v in v]) < 2
        ]
        class_configs = [
            {"shape": v[0], "color": v[1], "texture": v[2]} for v in class_configs
        ]
    else:
        raise ValueError(f"Dataset name not recognised: {name}")

    allowed = {
        "shapes": allowed_shapes,
        "colors": allowed_colors,
        "textures": allowed_textures
    }

    return class_configs, allowed
