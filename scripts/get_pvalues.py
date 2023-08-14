from pathlib import Path
from typing import Literal, Union

import pandas as pd
from scipy import stats


def get_df(csv_path: Union[str, Path]):
    df = pd.read_csv(
        csv_path, usecols=["filename", "dice", "iou", "hausdorff_distance"]
    )
    df.filename = df.filename.apply(lambda x: "/".join(x.rsplit("/", 2)[1:]))
    return df.set_index("filename", verify_integrity=True).sort_index()


def main(
    first_csv: Path,
    second_csv: Path,
    test_type: Literal["ttest", "wilcoxon"],
    alternative: Literal["two-sided", "greater", "less"],
    threshold: float,
    verbose: bool,
):
    higher_df = get_df(first_csv)

    lower_df = get_df(second_csv)

    print("\nFor first_csv:", first_csv)
    print("For second_csv:", second_csv)
    print(
        f"Applying: {test_type} paired test with H1: first_csv {alternative} than second_csv."
    )

    test_func = stats.ttest_rel if test_type == "ttest" else stats.wilcoxon

    dice_pvalue: float = test_func(
        higher_df.dice, lower_df.dice, alternative=alternative
    ).pvalue

    print(
        f"Pvalue for dice: {dice_pvalue}. (Reject H0: {dice_pvalue < threshold}. Th: {threshold})",
    )

    iou_pvalue: float = test_func(
        higher_df.iou, lower_df.iou, alternative=alternative
    ).pvalue

    print(
        f"Pvalue for IoU: {iou_pvalue}. (Reject H0: {iou_pvalue < threshold}. Th: {threshold})"
    )

    haus_alter = alternative

    if haus_alter == "less":
        haus_alter = "greater"
    elif haus_alter == "greater":
        haus_alter = "less"

    hausdorff_pvalue: float = test_func(
        higher_df.hausdorff_distance,
        lower_df.hausdorff_distance,
        alternative=haus_alter,
    ).pvalue

    print(
        f"Pvalue for Hausdorff distance: {hausdorff_pvalue}. (Reject H0: {hausdorff_pvalue < threshold}. Th: {threshold})"
    )

    if verbose:
        print("\nPrinting verbose stats:")

        dice_diff = higher_df.dice - lower_df.dice

        print("Dice diff mean:", dice_diff.mean())
        print("Dice diff std:", dice_diff.std())

        iou_diff = higher_df.iou - lower_df.iou

        print("\nIoU diff mean:", iou_diff.mean())
        print("IoU diff std:", iou_diff.std())

        haussdorff_diff = higher_df.haussdorff_distance - lower_df.haussdorff_distance

        print("\nHausdorff Distance diff mean:", haussdorff_diff.mean())
        print("Hausdorff Distance diff std:", haussdorff_diff.std())


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser()

    parser.add_argument(
        "--first-csv",
        type=Path,
        required=True,
        help="Path to the csv file with first set of dice and IoU values",
    )

    parser.add_argument(
        "--second-csv",
        type=Path,
        required=True,
        help="Path to the csv file with second set of dice and IoU values",
    )

    parser.add_argument(
        "--test-type",
        default="ttest",
        choices=("ttest", "wilcoxon"),
        help="Type of the test to use from scipy.stats package",
    )

    parser.add_argument(
        "--alternative",
        default="greater",
        choices=("two-sided", "greater", "less"),
        help="Alternative hypothesis for the t-test",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="The threshold for p-value to reject the null hypothesis",
    )

    parser.add_argument(
        "--verbose",
        action=BooleanOptionalAction,
        default=False,
        help="Print mean and std of the dice and IoU differences",
    )

    args = parser.parse_args()

    main(**vars(args))
