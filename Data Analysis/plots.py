import pandas as pd
import plotly.graph_objs as go
import sys
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.stats as stats
import pingouin as pg

PERCENTAGES = [25, 35, 45, 55, 65, 75]
color_encodings = { 'areas': '#f47068', 'textures': '#ffb3ae', 'lines': '#06d6a0', 'position_aligned': '#1697a6', 'position_unaligned': '#0e606b', 'curvatures': '#ffc24b'}
plotly_color_encodings = {'table': 'green', 'areas': 'red', 'textures': 'yellow', 'lines': 'blue', 'position_aligned': 'purple', 'position_unaligned': 'orange', 'curvatures': 'brown'}

# Create an ExcelFile object
xls = pd.ExcelFile('./data.xlsx')
file_path = './data.xlsx'

# Get a list of sheet names in the Excel file
sheet_names = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12']


def data_process(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Add participant ID column
    df['participant_id'] = sheet_name

    # divide the dataset into two    
    tc = df[['Tactile', 'Responded Accuracy', 'participant_id', 'Task 1']]
    vs = df[['Visual', 'Responded Accuracy.1', 'participant_id', 'Task 1.1']]

    # rename the columns respectively
    tc = tc.rename(columns={'Tactile': 'encoding', 'Responded Accuracy': 'RA', 'Task 1': 'FCWrong'})
    vs = vs.rename(columns={'Visual': 'encoding', 'Responded Accuracy.1': 'RA', 'Task 1.1': "FCWrong"})

    # Use regular expression to match the pattern and extract the relevant parts
    tc[['type', 'label']] = tc['encoding'].str.extract(r'(\w+)_(\d+)')
    vs[['type', 'label']] = vs['encoding'].str.extract(r'(\w+)_(\d+)')

    # Convert the 'suffix' column to integers
    tc['label'] = tc['label'].astype(int)
    vs['label'] = vs['label'].astype(int)

    # create a dictionary for true percentage
    ratios = {0: 25, 1: 35, 2: 45, 3: 55, 4: 65, 5: 75}

    # map the key to true percentage
    tc['truePercent'] = tc['label'].map(ratios)
    vs['truePercent'] = vs['label'].map(ratios)

    # make a column whose value is |responded acc - true acc|
    tc['abs_diff'] = (tc['RA'] -tc['truePercent']).abs()
    vs['abs_diff'] = (vs['RA'] -vs['truePercent']).abs()

    # drop unnecessary columns
    tc = tc.drop(['label', "encoding"], axis=1)
    vs = vs.drop(['label', "encoding"], axis=1)

    # divide dataset by type
    tc_pu = tc[tc['type'] == 'position_unaligned']
    vs_pu = vs[vs['type'] == 'position_unaligned']
    tc_pa = tc[tc['type'] == 'position_aligned']
    vs_pa = vs[vs['type'] == 'position_aligned']
    tc_a = tc[tc['type'] == 'areas']
    vs_a = vs[vs['type'] == 'areas']
    tc_l = tc[tc['type'] == 'lines']
    vs_l = vs[vs['type'] == 'lines']
    tc_c = tc[tc['type'] == 'curvatures']
    vs_c = vs[vs['type'] == 'curvatures']
    tc_t = tc[tc['type'] == 'textures']
    vs_t = vs[vs['type'] == 'textures']

    tactile = [tc_pu, tc_pa, tc_a, tc_l, tc_c, tc_t]
    visual = [vs_pu, vs_pa, vs_a, vs_l, vs_c, vs_t]

    # Add code to process "tabular condition"
    tab = df[['Tabular', 'Responded Accuracy.2', 'participant_id', 'Task 1.2']]
    tab = tab.rename(columns={'Tabular': 'encoding', 'Responded Accuracy.2': 'RA', 'Task 1.2': 'FCWrong'})
    tab[['type', 'label']] = tab['encoding'].str.extract(r'(\w+)_(\d+)')
    tab = tab.head(6)
    tab['label'] = tab['label'].astype(int)
    tab['truePercent'] = tab['label'].map(ratios)
    tab['abs_diff'] = (tab['RA'] - tab['truePercent']).abs()
    tab = tab.drop(['label', "encoding"], axis=1)
    tabular = [tab]

    return visual, tactile, tabular

def draw_visual_lines(deg):
    visual_data = combine_sheets("visual")

    # Plot the data
    fig, ax = plt.subplots(figsize=(6,8))

    for type in visual_data.keys():
        data_list = visual_data[type]

        combined = pd.concat(visual_data[type])

        grouped_diff = combined.groupby('truePercent')
        diff_mean = grouped_diff['abs_diff'].mean()


        # Fit a quadratic curve to the data
        coeffs = np.polyfit(PERCENTAGES, diff_mean.values, deg)
        quad_func = np.poly1d(coeffs)

        # Generate new x-values for the curve
        x_new = np.linspace(25, 75, 6)

        # Calculate y-values for the curve
        y_new = quad_func(x_new)

        ax.plot(x_new, y_new, color_encodings[type],label=type)
    
    # ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    ax.set_ylabel('Absolute Error')
    ax.set_xlabel('True Percentage Difference')
    ax.set_title('Visual Condition')
    plt.ylim(0, 25)
    plt.show()
        
def draw_visual_lines_plotly():
    visual_data = combine_sheets("visual")

    traces = []

    for type in visual_data.keys():
        data_list = visual_data[type]

        combined = pd.concat(visual_data[type])

        grouped_diff = combined.groupby('truePercent')
        diff_mean = grouped_diff['abs_diff'].mean()

        trace = go.Scatter(x=PERCENTAGES, y=diff_mean.values, mode="lines+markers", line_shape="linear", name=type, line=dict(color=plotly_color_encodings[type])) 


        traces.append(trace)

        
    # Create the figure and layout
    fig = go.Figure(data=traces, layout=go.Layout(title='Visual Encodings'))

    fig.update_layout(height=800, width=600)

    # Show the figure
    fig.show()
    
    
def draw_tactile_lines(deg):
    tactile_data = combine_sheets("tactile")

    # Plot the data
    fig, ax = plt.subplots(figsize=(6,8))

    for type in tactile_data.keys():
        data_list = tactile_data[type]

        combined = pd.concat(tactile_data[type])

        grouped_diff = combined.groupby('truePercent')
        diff_mean = grouped_diff['abs_diff'].mean()


        # Fit a quadratic curve to the data
        coeffs = np.polyfit(PERCENTAGES, diff_mean.values, deg)
        quad_func = np.poly1d(coeffs)

        # Generate new x-values for the curve
        x_new = np.linspace(25, 75, 6)

        # Calculate y-values for the curve
        y_new = quad_func(x_new)

        ax.plot(x_new, y_new, color_encodings[type], label=type)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Absolute Error')
    ax.set_xlabel('True Percentage Difference')
    ax.set_title('Tactile Condition')
    plt.ylim(0, 25)
    plt.show()


def tacVsVis():
    tactile_data = combine_sheets('tactile')
    visual_data = combine_sheets('visual')
    tabular_data = combine_sheets('tabular')

    tactile_total = pd.DataFrame()
    visual_total = pd.DataFrame()
    tabular_total = pd.DataFrame()


    for type in tactile_data.keys():
        tactile_combined = pd.concat(tactile_data[type])
        tactile_total = pd.concat([tactile_total, tactile_combined])

        visual_combined = pd.concat(visual_data[type])
        visual_total = pd.concat([visual_total, visual_combined])

    tabular_combined = pd.concat(tabular_data['table'])
    tabular_total = pd.concat([tabular_total, tabular_combined])

    # using matplotlib
    box_data = [visual_total['abs_diff'], tactile_total['abs_diff'], tabular_total['abs_diff']]

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the boxplots
    ax.boxplot(box_data)

    # Add labels and title
    ax.set_xticklabels(['Visual', 'Tactile', 'Tabular'])
    ax.set_ylabel('Absolute Error')
    ax.set_title('Multiple Boxplots')

    # Show the plot
    plt.show()

    # Calculate the means
    visual_mean = visual_total['abs_diff'].mean()
    tactile_mean = tactile_total['abs_diff'].mean()
    tabular_mean = tabular_total['abs_diff'].mean()

    # Print the means
    print(f"Mean Absolute Error for Visual: {visual_mean}")
    print(f"Mean Absolute Error for Tactile: {tactile_mean}")
    print(f"Mean Absolute Error for Tabular: {tabular_mean}")


    tabular_total.to_excel('table_data.xlsx', index=False)
    # # using Plotly
    # box_traces = []
    # box_traces.append(go.Box(y=visual_total['abs_diff'], name='Visual'))
    # box_traces.append(go.Box(y=tactile_total['abs_diff'], name='Tactile'))

    # # Create a layout object
    # layout = go.Layout(title='Visual vs. Tactile')

    # # Create a figure object
    # fig = go.Figure(data=box_traces, layout=layout)

    # # Show the plot
    # fig.show()

def anovatest():
    tactile_data = combine_sheets('tactile')
    visual_data = combine_sheets('visual')

    tactile_total = pd.DataFrame()
    visual_total = pd.DataFrame()

    i = 0
    for type in tactile_data.keys():
        tactile_combined = pd.concat(tactile_data[type])
        # tactile_total = tactile_combined['abs_diff_2']
        tactile_total = pd.concat([tactile_total, tactile_combined])


        visual_combined = pd.concat(visual_data[type])
        # visual_total = visual_combined['abs_diff']
        visual_total = pd.concat([visual_total, visual_combined])

    # using matplotlib
    box_data = [visual_total['abs_diff'], tactile_total['abs_diff']]

    F_statistic, p_value = stats.f_oneway(box_data[0], box_data[1])

    # Print results
    print("F-statistic:", F_statistic)
    print("P-value:", p_value)

def rm_anova_test():
    tactile_data = combine_sheets('tactile')
    visual_data = combine_sheets('visual')

    tactile_total = pd.DataFrame()
    visual_total = pd.DataFrame()

    for type in tactile_data.keys():
        tactile_combined = pd.concat(tactile_data[type])
        tactile_total = pd.concat([tactile_total, tactile_combined])

        visual_combined = pd.concat(visual_data[type])
        visual_total = pd.concat([visual_total, visual_combined])

    tactile_total['condition'] = 'Tactile'
    visual_total['condition'] = 'Visual'

    combined_data = pd.concat([visual_total, tactile_total], ignore_index=True)
    combined_data = combined_data.rename(columns={'abs_diff': 'abs_error'})
    combined_data.to_excel('my_data.xlsx', index=False)

    # Perform the repeated measures ANOVA
    aov = pg.rm_anova(dv='abs_error', within='condition', subject='participant_id', data=combined_data)

    print(aov)

     # Perform the post-hoc test if the p-value is less than the significance level
    significance_level = 0.05
    if aov.at[0, 'p-unc'] < significance_level:
        posthoc = pg.pairwise_tests(dv='abs_error', within='condition', subject='participant_id', data=combined_data)
        print("\nPost-hoc Test:")
        print(posthoc)
    else:
        print("Post-hoc test not applicable: p-value >= significance level")

def compare_accuracy():
    visual_total = pd.DataFrame()
    tactile_total = pd.DataFrame()

    for sheet_name in sheet_names:
        visual_dat, tactile_dat = data_process(file_path, sheet_name)
        visual_combined = pd.concat(visual_dat)
        tactile_combined = pd.concat(tactile_dat)
        visual_total = pd.concat([visual_total, visual_combined])
        tactile_total = pd.concat([tactile_total, tactile_combined])

    visual_correct = visual_total[visual_total['FCWrong'].isna()]
    tactile_correct = tactile_total[tactile_total['FCWrong'].isna()]

    visual_accuracy = len(visual_correct) / len(visual_total)
    tactile_accuracy = len(tactile_correct) / len(tactile_total)

    print("Mean accuracy for Visual condition: {:.2f}%".format(visual_accuracy * 100))
    print("Mean accuracy for Tactile condition: {:.2f}%".format(tactile_accuracy * 100))

    visual_total['condition'] = 'Visual'
    tactile_total['condition'] = 'Tactile'

    combined_data = pd.concat([visual_total, tactile_total], ignore_index=True)
    combined_data['accuracy'] = np.where(combined_data['FCWrong'].isna(), 1, 0)

    aov = pg.rm_anova(dv='accuracy', within='condition', subject='participant_id', data=combined_data)
    print("\nRepeated Measures ANOVA Test Results:")
    print(aov)



def visual_encodings():
    visual_data = combine_sheets('visual')

    # Create a figure and axis object
    fig, ax = plt.subplots()

    i = 0
    for type in visual_data.keys():
        combined = pd.concat(visual_data[type])
        ax.boxplot(combined['abs_diff'], positions=[i+1], labels=[type], vert=False)
        i += 1

    # Add labels and title
    ax.set_ylabel('Encodings')
    ax.set_xlabel('Absolute Error')
    ax.set_title('Visual Encodings')

    plt.show()


def tactile_encodings():
    tactile_data = combine_sheets('tactile')

    # Create a figure and axis object
    fig, ax = plt.subplots()

    i = 0
    for type in tactile_data.keys():
        combined = pd.concat(tactile_data[type])
        ax.boxplot(combined['abs_diff'], positions=[i+1], labels=[type], vert=False)
        i += 1

    # Add labels and title
    ax.set_ylabel('Encodings')
    ax.set_xlabel('Absolute Error')
    ax.set_title('Tactile Encodings')

    plt.show()

def rm_anova_test_by_condition(condition):
    data = combine_sheets(condition)
    data_total = pd.DataFrame()

    for encoding_type in data.keys():
        combined = pd.concat(data[encoding_type])
        data_total = pd.concat([data_total, combined])

    data_total['encoding'] = data_total['type']
    data_total = data_total.rename(columns={'abs_diff': 'abs_error'})

    # Perform the repeated measures ANOVA
    aov = pg.rm_anova(dv='abs_error', within='encoding', subject='participant_id', data=data_total)

    print(f"Results for {condition.capitalize()} Condition:")
    print(aov)
    print("\n")

    # Calculate and print the mean for each encoding type
    print("Mean for each encoding type:")
    encoding_means = data_total.groupby('encoding')['abs_error'].mean()
    print(encoding_means)
    print("\n")

    # Perform the post-hoc test if the p-value is less than the significance level
    significance_level = 0.05
    if aov.at[0, 'p-unc'] < significance_level:
        posthoc = pg.pairwise_tests(dv='abs_error', within='encoding', subject='participant_id', data=data_total)
        print("Post-hoc Test:")
        print(posthoc)
    else:
        print("Post-hoc test not applicable: p-value >= significance level")


def combine_sheets(type):
    visual_encodings = {key: [] for key in ['curvatures', 'lines', 'areas', 'textures', 'position_aligned','position_unaligned']}
    tactile_encodings = {key: [] for key in ['curvatures', 'lines', 'areas', 'textures', 'position_aligned','position_unaligned']}
    tabular_encodings = {key: [] for key in ['table']}


    visual = []
    tactile = []
    tabular = []
    num_type = 6

    for sheet_name in sheet_names:
        visual_dat, tactile_dat, tab_dat = data_process(file_path, sheet_name)

        visual.append(visual_dat)
        tactile.append(tactile_dat)
        tabular.append(tab_dat)
    

    num_data = len(visual)
    if type == 'tabular':
        num_data = len(tabular)
        num_type = 1

    for j in range(num_data):
        for i in range(num_type):
            type_name = visual[j][i]["type"].iloc[0]
            if type == "visual":
                # visual_encodings[type_name].append((visual[j][i]).drop('type', axis=1))
                visual_encodings[type_name].append((visual[j][i]))
            elif type == 'tactile':
                # tactile_encodings[type_name].append((tactile[j][i]).drop('type', axis=1))
                tactile_encodings[type_name].append((tactile[j][i]))
            elif type == 'tabular':
                # tactile_encodings[type_name].append((tactile[j][i]).drop('type', axis=1))
                tabular_encodings['table'].append((tabular[j][i]))
    
    if type == "visual":
        return visual_encodings
    elif type == 'tactile':
        return tactile_encodings
    elif type == 'tabular':
        return tabular_encodings

arg = sys.argv[1]
if len(sys.argv) == 3:
    deg = int(sys.argv[2])

if arg == 'visual_lines':
    draw_visual_lines(deg)
elif arg == 'tactile_lines':
    draw_tactile_lines(deg)
elif arg == 'tacVsVis':
    tacVsVis()
elif arg == 'visualEnc':
    visual_encodings()
elif arg == 'tactileEnc':
    tactile_encodings()
elif arg == 'visLinesPlotly':
    draw_visual_lines_plotly()
elif arg == 'anova':
    anovatest()
elif arg == 'rm_anova':
    rm_anova_test()
elif arg == 'rm_anova_visual':
    rm_anova_test_by_condition('visual')
elif arg == 'rm_anova_tactile':
    rm_anova_test_by_condition('tactile')
elif arg == 'compare_acc':
    compare_accuracy()
