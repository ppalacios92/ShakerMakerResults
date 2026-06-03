# Plotting

All plotting functions. The single-model functions are also exposed as methods
on `ShakerMakerData`.

!!! note "Why the signatures show `self`"
    The single-model plotting functions below are written as module-level
    functions whose **first argument is the model itself** — they are attached
    to `ShakerMakerData` as methods. In practice you call them *without* `self`:

    ```python
    model_window.plot_node_response(node_id=10, data_type="vel")
    # equivalent to:
    # plot_node_response(model_window, node_id=10, data_type="vel")
    ```

    The recurring inputs (`data_type`, `component`, `factor`, `node_id` /
    `target_pos`, …) are explained once in the
    [Common parameters glossary](../guides/parameters.md).

## Single model — domain

::: ShakerMakerResults.plotting.single_model.domain_plots

## Single model — nodes

::: ShakerMakerResults.plotting.single_model.node_plots

## Single model — surface

::: ShakerMakerResults.plotting.single_model.surface_plots

## Single model — animations

::: ShakerMakerResults.plotting.single_model.animation_plots

## Single model — maps

::: ShakerMakerResults.plotting.single_model.map_plots

## Comparison — response & spectra

::: ShakerMakerResults.plotting.comparison.response_plots

## Comparison — Green's Functions

::: ShakerMakerResults.plotting.comparison.gf_plots

## Comparison — domains

::: ShakerMakerResults.plotting.comparison.domain_plots

## Comparison — Arias

::: ShakerMakerResults.plotting.comparison.arias_plots
