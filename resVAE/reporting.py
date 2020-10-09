#Copyright (C) 2019  Soeren Lukassen

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

import base64
from io import BytesIO

import seaborn as sns
from matplotlib import pyplot as plt

from .utils import *


def _display_side_by_side(df: pd.DataFrame, split: int = 20):
    """
    Utility function to place several tables based on pandas DataFrames side by side in an html file. Loosely based on the discussion at https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side

    :param df: pandas DataFrame to be displayed as split html table.
    :param split: Integer value indicating the maximum row count per table before a split is applied (default: 20)
    :return: returns a string holding an HTML representation of the DataFrame
    """
    html_str = ''
    if df.shape[0] > split:
        for i in range(int(np.floor(df.shape[0]/split))):
            iter_low = i * 20
            iter_high = (i + 1) * 20
            iter_df = df.iloc[iter_low:iter_high, :]
            iter_df.columns = ['Ranks '+str(iter_low)+' - '+str(iter_high)]
            html_str += iter_df.to_html()
        iter_df = df.iloc[iter_high:, :]
        iter_df.columns = ['Ranks '+str(iter_low)+' - '+str(df.shape[0])]
    else:
        iter_df = df
    html_str += iter_df.to_html()
    html_str = html_str.replace('table ', 'table style="display:inline"')
    html_str = html_str.replace('border="1"', '')
    return html_str


def _weightplot_str(weights: pd.DataFrame, cluster: int = 0):
    """
    Generate a lineplot showing weights vs ranks for a neuron, and return the result as an svg string.

    :param weights: pandas DataFrame to be displayed as split html table
    :param cluster: integer indicating the column of the pandas DataFrame to be used for plotting
    :return: a string containing the svg image
    """
    pos_cutoff = calculate_elbow(weights.iloc[cluster, :])
    neg_cutoff = calculate_elbow(weights.iloc[cluster, :], negative=True)
    plt.figure(figsize=(8, 3))
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.plot(np.sort(weights.iloc[cluster, :]))
    plt.vlines([pos_cutoff, neg_cutoff],
               ymin=np.min(weights.iloc[cluster, :]),
               ymax=np.max(weights.iloc[cluster, :]),
               linestyles='dotted')
    plt.title('Gene weight cutoffs - Cluster '+str(cluster))
    plt.xlabel('Gene rank')
    plt.ylabel('Weight')
    figfile = BytesIO()
    plt.savefig(figfile, format='svg', transparent=True)
    plt.close()
    figfile.seek(0)  # rewind to beginning of file
    return figfile.read().decode('UTF8').replace('\n*{stroke-linecap:butt;stroke-linejoin:round;white-space:pre;}', '')


def _clustermap_str(weights: pd.DataFrame):
    """
    Generate a clustered heatmap of the neuron weight mappings as string representation of a png

    :param weights: pandas DataFrame to be displayed as clustered heatmap
    :return: a string containing the png image
    """
    sns.clustermap(weights, row_cluster=False, standard_scale=None)
    figfile = BytesIO()
    plt.savefig(figfile, format='png', transparent=True, dpi=96)
    plt.close()
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png.decode('UTF8')


def _report_block_template(weights: pd.DataFrame, cluster: int = 0):
    """
    Generate a block of HTML code containing the table and images for an individual neurons weight mappings.

    :param weights: pandas DataFrame containing the weight mappings
    :param cluster: integer indicating the column of the pandas DataFrame to be used for report generation
    :return: a string containing an HTML representation of a report block
    """
    image = _weightplot_str(weights, cluster)
    image = image.replace('width="576pt"', 'width="80%"')
    cluster = cluster
    pos_cutoff = calculate_elbow(weights.iloc[cluster, :])
    neg_cutoff = calculate_elbow(weights.iloc[cluster, :], negative=True)
    negative = weights.iloc[cluster, np.argsort(weights.iloc[cluster, :])][:neg_cutoff]
    positive = weights.iloc[cluster, np.argsort(weights.iloc[cluster, :])][pos_cutoff:]
    report_block = ('<body>' +
                    '<h2>Cluster ' + str(cluster) + '</h2>' +
                    '<div width="100%" align="justify">' +
                    image +
                    '</div>' +
                    '<br>' +
                    '<h3>Positive enrichment:</h3>' +
                    _display_side_by_side(pd.DataFrame(positive).iloc[::-1]) +
                    '<br>' +
                    '<h3>Negative enrichment:</h3>' +
                    _display_side_by_side(pd.DataFrame(negative)) +
                    '<br>' +
                    '<hr>' +
                    '</body>')
    return report_block


def _report_head_template(weights: pd.DataFrame):
    """
    Generate a block of HTML code containing the header for a resVAE report.

    :param weights: pandas DataFrame containing the weight mappings
    :return: a string containing an HTML representation of a report header
    """
    clustmap = _clustermap_str(weights)
    head_block = ('' +
                  '<!DOCTYPE html>\n' +
                  '<html lang="en">\n' +
                  '<head>\n' +
                  '<meta charset="utf-8">\n' +
                  '<title>resVAE report</title>\n' +
                  '<link rel="stylesheet" href="style.css" type="text/css">\n' +
                  '<h1>resVAE Report</h1>' +
                  '<img src="data:image/png;base64,' + clustmap + '"\ >' +
                  '<hr>'
                  '</head>\n')
    return head_block


def generate_html_report(weights: pd.DataFrame, path: str, neurons_use=None):
    """
    Generates an HTML report for the resVAE results representing the weight mapping of an individual layers, and writes this to file.

    :param weights: pandas DataFrame containing the weight mappings
    :param path: path to the output file
    :param neurons_use: Which neurons to generate the report for. None type indicates a report for all neurons of this layer
    :return: None
    """
    if not os.path.isdir(os.path.split(path)[0]):
        os.mkdir(os.path.split(path)[0])
    static_report = ''
    static_report += _report_head_template(weights)
    if neurons_use is None:
        for neurons in range(weights.shape[0]):
            _static_block = _report_block_template(weights, cluster=neurons)
            static_report += _static_block
    else:
        for neurons in neurons_use:
            _static_block = _report_block_template(weights, cluster=neurons)
            static_report += _static_block
    html_file = open(path, "w")
    html_file.write(static_report)
    html_file.close()
    return None
