import os
from typing import Optional

from matplotlib import pyplot as plt


def save_or_show_plot(output_file: Optional[str] = None) -> None:
    """
    Saves the plot to a PDF file if an output file path is provided, or displays the plot on screen if no file path is specified.

    Parameters:
        output_file (Optional[str]): The file path where the plot will be saved as a PDF. If None, the plot will be displayed using plt.show().

    Returns:
        None
    """
    if output_file is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save the figure as a PDF
        plt.savefig(output_file)
        plt.close()  # Close the plot to free up memory
    else:
        # Display the plot
        plt.show()