import itertools

from elements.classes import ElementDataset


class Element(ElementDataset):
    def __init__(self, dataset_name, **kwargs):
        class_configs, allowed, allowed_combinations = select_dataset(dataset_name)
        super().__init__(
            allowed, class_configs, allowed_combinations=allowed_combinations, **kwargs
        )

    def __getitem__(self, idx):
        item = self.get_item(idx)
        img = self.transform(item.img)
        return {"image": img.permute(1, 2, 0), "class_label": item.class_labels_oh}


def select_dataset(name):
    # By default, allow all combinations of shape, color, texture in the dataset
    # and the "simple" set of concepts
    allowed_combinations = None
    allowed_shapes = ["square", "circle", "triangle", "plus"]
    allowed_colors = ["red", "green", "blue"]
    allowed_textures = ["solid", "spots_polka", "stripes_diagonal"]

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
    elif name == "simple_all":
        class_configs = select_all_combinations_of_classes(
            allowed_shapes, allowed_colors, allowed_textures
        )

    elif name == "simple_all_non_overlapping":
        class_configs = select_all_combinations_of_classes(
            allowed_shapes, allowed_colors, allowed_textures
        )

        # Restrict some combinations of concepts from appearing in the dataset
        restrictions = [
            (None, "red", "stripes_diagonal"),
            ("triangle", "green", None),
            ("plus", None, "spots_polka"),
            ("circle", None, "solid"),
        ]
        allowed_combinations = list(
            itertools.product(allowed_shapes, allowed_colors, allowed_textures)
        )
        allowed_combinations = remove_matching_items(allowed_combinations, restrictions)

        # Remove classes which will no longer appear
        class_configs_tuples = [
            tuple([in_v for in_v in v.values()]) for v in class_configs
        ]
        class_configs_tuples = remove_matching_items(class_configs_tuples, restrictions)
        class_configs = [
            {"shape": v[0], "color": v[1], "texture": v[2]}
            for v in class_configs_tuples
        ]
    else:
        raise ValueError(f"Dataset name not recognised: {name}")

    allowed = {
        "shapes": allowed_shapes,
        "colors": allowed_colors,
        "textures": allowed_textures,
    }

    return class_configs, allowed, allowed_combinations


def select_all_combinations_of_classes(
    allowed_shapes, allowed_colors, allowed_textures
):
    allowed_config = [
        v + [None] for v in [allowed_shapes, allowed_colors, allowed_textures]
    ]
    class_configs = list(itertools.product(*allowed_config))

    class_configs = [v for v in class_configs if sum([in_v is None for in_v in v]) < 2]
    class_configs = [
        {"shape": v[0], "color": v[1], "texture": v[2]} for v in class_configs
    ]
    return class_configs


def remove_matching_items(combinations, restrictions):
    filtered_arr = []
    for combo in combinations:
        match_found = False
        for restriction in restrictions:
            if all(x is None or x == y for x, y in zip(restriction, combo)):
                match_found = True
                break
        if not match_found:
            filtered_arr.append(combo)
    return filtered_arr
