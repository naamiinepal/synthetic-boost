import json
from glob import glob
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def get_metrics(
    glob_pattern: str,
    prompts: Union[List[str], Tuple[str]],
    model_name: str,
    stage: str,
    encoder_status: str,
):
    metric_files = glob(glob_pattern)
    print(metric_files)

    assert len(metric_files) == len(
        prompts
    ), f"The number of prompts: {len(prompts)} and csv files: {len(metric_files)} should be equal"

    metric_files.sort()

    dataframes = []

    for file, p in zip(metric_files, prompts):
        df = pd.read_csv(file)
        df["prompt_type"] = p

        dataframes.append(df)

    concat_df = pd.concat(dataframes, ignore_index=True)

    concat_df["model_name"] = model_name
    concat_df["stage"] = stage
    concat_df["encoder_status"] = encoder_status

    return concat_df


def main(
    config_path: Path,
    output_dir: Path,
    metric: str,
    save_fmt: str,
    show_yticks: bool,
    yrange: Tuple[float, float],
    legend_location: str,
    x_label_size: int,
    y_label_size: int,
    linewidth: float,
    plot_type: str,
    base_status: str,
    model: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        plots = json.load(f)

    # get only the json with models
    plots = [p for p in plots if p["model_name"] == model]

    plot_nums = len(plots)

    prompts_collection = [[f"p{i}" for i in range(p["prompt_number"])] for p in plots]

    glob_patterns = [plot["csv_glob"] for plot in plots]

    assert (
        len(glob_patterns) == plot_nums
    ), "The glob patterns must be equal to the prompt numbers"

    model = [plot["model_name"] for plot in plots]

    assert len(model) == plot_nums, "The models must be equal to the prompt numbers"

    stages = [plot["stage"] for plot in plots]

    assert len(stages) == plot_nums, "The stages must be equal to the prompt numbers"

    encoder_status = [plot["encoder"] for plot in plots]
    assert (
        len(encoder_status) == plot_nums
    ), "The encoder status must be equal to the prompt numbers"

    concat_df = pd.concat(
        [
            get_metrics(*args)
            for args in zip(
                glob_patterns, prompts_collection, model, stages, encoder_status
            )
        ],
        ignore_index=True,
    )

    # concat_df["dice"] = concat_df["dice"]

    sns.set_palette(sns.color_palette("tab10"))
    if args.plot_type == "diff":
        concat_df = concat_df[
            [metric, "stage", "prompt_type", "model_name", "encoder_status"]
        ]
        mean_df = (
            concat_df.groupby(["prompt_type", "model_name", "stage", "encoder_status"])
            .mean()
            .reset_index()
            .sort_values(by=["prompt_type", "model_name"])
        )
        mean_df["diff"] = (
            mean_df.groupby(["prompt_type", "model_name", "stage"])[metric]
            .diff()
            .fillna(0)
        )
        sns.set_style("whitegrid")

        # remove the base status
        mean_df = mean_df[mean_df["encoder_status"] != base_status]

        metric_plot = sns.lineplot(
            data=mean_df,
            x="prompt_type",
            y="diff",
            hue="stage",
            style="encoder_status",
            linewidth=linewidth,
            markers=True,
            markersize=10,
        )
        metric_plot.axes.spines["bottom"].set_position("zero")
        metric_plot.set_ylabel(f"Difference in {metric.capitalize()}")
        metric_plot.set_xlabel("")

    else:
        metric_plot = sns.lineplot(
            data=concat_df,
            x="prompt_type",
            y=metric,
            hue="stage",
            style="encoder_status",
            markers=True,
            linewidth=linewidth,
            legend="auto" if legend_location == "outside" else legend_location,
        )

    if yrange:
        metric_plot.set(ylim=yrange)
    if legend_location == "outside":
        metric_plot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    metric_plot.tick_params(axis="x", labelsize=x_label_size)
    metric_plot.tick_params(axis="y", labelsize=y_label_size)

    if not show_yticks:
        metric_plot.set(yticks=[])
        metric_plot.set(ylabel=None)

    model = "_".join(sorted(set(model)))

    fig_name = f"{config_path.stem}_{metric}_{model}_{plot_type}_base_{base_status}"

    fig: Figure = metric_plot.get_figure()
    fig.savefig(output_dir / f"{fig_name}.{save_fmt}", bbox_inches="tight")
    print(f"Saved figure to {output_dir / f'{fig_name}.{save_fmt}'}")


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser()

    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to the config file containing the glob patterns, model names and stages",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the output directory where the plots will be saved",
    )

    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="The metric to plot. Must be one of iou, dice, surface_dice, hausdorff_distance",
    )

    parser.add_argument(
        "--save-fmt",
        type=str,
        default="png",
        help="The format to save the plots in. Must be one of png, pdf, svg",
    )

    parser.add_argument(
        "--show-yticks",
        action=BooleanOptionalAction,
        help="Whether to show the yticks or not",
    )

    parser.add_argument(
        "--yrange",
        type=float,
        nargs=2,
        default=None,
        help="The range of the y-axis",
    )

    parser.add_argument(
        "--legend-location",
        type=str,
        default=None,
        help="The location of the legend",
    )

    parser.add_argument(
        "--x-label-size",
        type=int,
        default=10,
        help="The size of the x-axis labels",
    )

    parser.add_argument(
        "--y-label-size",
        type=int,
        default=5,
        help="The size of the y-axis labels",
    )

    parser.add_argument(
        "--linewidth",
        type=float,
        default=2.5,
        help="The width of the line",
    )

    parser.add_argument(
        "--plot-type",
        type=str,
        default="line",
        help="The type of plot to be used. Must be one of line, diff",
    )

    parser.add_argument(
        "--base-status",
        type=str,
        default="freeze",
        help="The base encoder status to be used for the diff plot. Must be one of freeze, unfreeze.",
    )

    parser.add_argument(
        "--model", type=str, help="Models to use for the plot", required=True
    )

    args = parser.parse_args()

    main(**vars(args))
