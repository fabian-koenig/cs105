## Analyzes readability of privacy policies
# Import packages
import matplotlib.pyplot as plt
import os
import pandas as pd
import researchpy as rp
import scipy.stats as stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
import string
import textstat

def main():
    # For each text file in folder with privacy policies, store statistics in pandas dataframe
    columnNames = ['source', 'industry', 'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'dale_chall', 'reading_time']
    policies_df = pd.DataFrame(columns=columnNames)
    # Industry codes used to refer to directory folders
    naics = [11, 21, 22, 23, 31, 42, 44, 48, 51, 52, 53, 54, 55, 61, 62, 71, 72]
    industry_names = ['Agriculture', 'Mining', 'Utilities', 'Construction', 'Manufacturing', 'Wholesale Trade', 'Retail Trade',
                      'Transportation', 'Information', 'Finance and Insurance', 'Real Estate',
                      'Professional, Scientific, and Technical Services', 'Management of Companies and Enterprises',
                      'Educational Services', 'Health Care', 'Arts, Entertainment, and Recreation', 'Accommodation and Food Services']

    ## Compiling relevant metrics into dataframe
    # For each industry, read in the privacy policy in txt format
    for code in naics:
        path = os.path.join("/Users/fabiankoenig/OneDrive - Harvard University/College/CS105/3_Your rights at Harvard/pp_txtfiles/", str(code))
        os.chdir(path)
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".txt"):
                file_path = f"{path}/{file}"
                # Empty list for row values in dataframe
                values = []
                # Call read text file function
                with open(file_path, 'r') as f:
                    # Add name of company
                    head, tail = os.path.split(file_path)
                    values.append(tail[0:-4])
                    # Add industry code
                    values.append(code)

                    # Open privacy policy and add metrics
                    policy_text = f.read()
                    # Flesch Reading Ease Score (0-100)
                    values.append(textstat.flesch_reading_ease(policy_text))
                    # Flesch-Kincaid Grade Level
                    values.append(textstat.flesch_kincaid_grade(policy_text))
                    # Gunning FOG Formula
                    values.append(textstat.gunning_fog(policy_text))
                    # Dale-Chall Readability Test
                    values.append(textstat.dale_chall_readability_score(policy_text))
                    # Reading time
                    values.append(textstat.reading_time(policy_text, ms_per_char=14.69))

                    # Turn list of values into panda series and append to dataframe
                    values_series = pd.Series()
                    value_series = pd.Series(values, index=policies_df.columns)
                    policies_df = policies_df.append(value_series, ignore_index=True)

    ## Data Cleaning
    # Replace industry codes with industry names
    for (naics, name) in zip(naics, industry_names):
        replace_industry(naics, name, policies_df)

    # Convert reading time from seconds to minutes
    policies_df['reading_time'] = policies_df['reading_time'] / 60

    # Save dataframe as csv
    #policies_df.to_csv("~/pp_txtfiles/output/policies.csv", index=False)


    ## Statistics
    # Output descriptive statistics into text file
    text_file = ["ORDER: MEAN, MEDIAN, MAX, MIN, VARIANCE, STANDARD DEVIATION"]
    descriptives = ["flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog", "dale_chall", "reading_time"]

    # Add descriptives to list
    text_file = add_statistics(descriptives, text_file, policies_df)

    # Write statistics into textfile
    with open('/Users/fabiankoenig/OneDrive - Harvard University/College/CS105/3_Your rights at Harvard/pp_txtfiles/output/text/descriptives.txt', 'w') as f:
        for statistic in text_file:
            f.write(str(statistic) + "\n")

    ## Statistics
    # Triangulate Gunning Fog and Flesch-Kincaid Grade by determining correlation
    correlation = pearsonr(policies_df['gunning_fog'], policies_df['flesch_kincaid_grade'])
    # R2 = 0.99
    # p < 0.01

    # Determine whether industries differ significantly (ANOVA) - Length
    anova_values = pd.DataFrame()
    # Assess normal distribution
    for statistic in descriptives:
        unique_industries = policies_df['industry'].unique()
        for industry in unique_industries:
            stats.probplot(policies_df[policies_df['industry'] == industry][statistic], dist="norm", plot=plt)
            plt.title("Probability Plot - " +  industry)
            plt.show()
        # Data is normally distributed, can go forward with ANOVA
        mod = ols(str(statistic) + '~ industry', data=policies_df).fit()
        aov_table = sm.stats.anova_lm(mod, typ=2)
        aov_table['statistic'] = statistic
        anova_values = anova_values.append(aov_table)
    # no significant differences between industries

    ## Plots
    # Flesch Reading Ease Score (0-100)
    plot_boxplot(statistic='flesch_reading_ease', title='Flesch Reading Ease', xlabel='Score (0-100)', df=policies_df)
    # Flesch-Kincaid Grade
    plot_boxplot(statistic='flesch_kincaid_grade', title='Flesch-Kincaid Grade Level', xlabel='Grade (0-18+)', df=policies_df)
    # Gunning FOG Formula
    plot_boxplot(statistic='gunning_fog', title='Gunning Fog', xlabel='Index (6-17)', df=policies_df)
    # Dale-Chall Readability Test
    plot_boxplot(statistic='dale_chall', title='Dale-Chall Readability Test', xlabel='Score (-9.9)', df=policies_df)
    # Reading time
    plot_boxplot(statistic='reading_time', title='Reading Time', xlabel='Minutes', df=policies_df)

    # Readability by industry
    plot_barplot('flesch_reading_ease', 'Readability (Flesch Reading Ease) by Industry', 'Industry', 'Score (0-100)', policies_df)
    plot_barplot('flesch_kincaid_grade', 'Readability (Flesch-Kincaid Grade Level) by Industry', 'Industry', 'Grade (0-18+)', policies_df)
    plot_barplot('gunning_fog', 'Readability (Gunning Fog) by Industry', 'Industry', 'Index (6-17)', policies_df)
    plot_barplot('dale_chall', 'Readability (Dale-Chall Readability Test) by Industry', 'Industry', 'Score (-9.9)', policies_df)
    plot_barplot('reading_time', 'Reading Time', 'Industry', 'Minutes', policies_df)

    # Colorful readability / time scatter plot
    plot_scatter(xstatistic='reading_time', ystatistic='flesch_reading_ease', title='Readability (Flesch Reading Ease)', xlabel='Reading Time (in Minutes)', ylabel='Score', df=policies_df)
    plot_scatter(xstatistic='reading_time', ystatistic='flesch_kincaid_grade', title='Readability (Flesch-Kincaid Grade Level)', xlabel='Reading Time (in Minutes)', ylabel='Grade Level', df=policies_df)
    plot_scatter(xstatistic='reading_time', ystatistic='gunning_fog', title='Readability (Gunning Fog)', xlabel='Reading Time (in Minutes)', ylabel='Index', df=policies_df)
    plot_scatter(xstatistic='reading_time', ystatistic='dale_chall', title='Readability (Dale-Chall)', xlabel='Reading Time (in Minutes)', ylabel='Score', df=policies_df)

    plot_scatter_onlyindustry('reading_time', 'flesch_reading_ease', 'Readability (Flesch Reading Ease) by Industry', 'Reading Time (in Minutes)', 'Score', policies_df)


    ## Analyze historical trends
    # Create empty dataframe with columns
    columnNames = ['source', 'date', 'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'dale_chall', 'reading_time']
    historical_df = pd.DataFrame(columns=columnNames)

    # Read in privacy policies
    companies = ["boeing", "whatsapp", "google"]
    for company in companies:
        path = os.path.join("/Users/fabiankoenig/OneDrive - Harvard University/College/CS105/3_Your rights at Harvard"
                            "/pp_txtfiles/historical/", str(company))
        os.chdir(path)
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".txt"):
                file_path = f"{path}/{file}"
                values = []
                # call read text file function
                with open(file_path, 'r') as f:
                    head, tail = os.path.split(file_path)
                    values.append(company)
                    values.append(tail[0:-4])

                    policy_text = f.read()

                    # Flesch Reading Ease Score (0-100)
                    values.append(textstat.flesch_reading_ease(policy_text))

                    # Flesch-Kincaid Grade Level
                    values.append(textstat.flesch_kincaid_grade(policy_text))

                    # Gunning FOG Formula
                    values.append(textstat.gunning_fog(policy_text))

                    # Dale-Chall Readability Test
                    values.append(textstat.dale_chall_readability_score(policy_text))

                    # Reading time
                    values.append(textstat.reading_time(policy_text, ms_per_char=14.69))

                    values_series = pd.Series()
                    value_series = pd.Series(values, index=historical_df.columns)
                    historical_df = historical_df.append(value_series, ignore_index=True)

    ## Data cleaning
    # Convert reading time from seconds to minutes
    historical_df['reading_time'] = historical_df['reading_time'] / 60

    # Date formatting
    historical_df['date'] = pd.to_datetime(historical_df['date'])

    # Create subsets for data for each company
    google_sub = historical_df[historical_df['source'] == "google"]
    boeing_sub = historical_df[historical_df['source'] == "boeing"]
    whatsapp_sub = historical_df[historical_df['source'] == "whatsapp"]

    google_sub = google_sub.sort_values(by=['date'])
    boeing_sub = boeing_sub.sort_values(by=['date'])
    whatsapp_sub = whatsapp_sub.sort_values(by=['date'])

    ## READING TIME
    plot_historical('reading_time', google_sub, boeing_sub, whatsapp_sub, 'Historical Development of Privacy Policy Length', 'Reading Time (in Minutes)', 'Google', 'Boeing', 'WhatsApp')
    plot_historical('flesch_reading_ease', google_sub, boeing_sub, whatsapp_sub, 'Historical Development of Privacy Policy Readability', 'Score', 'Google', 'Boeing', 'WhatsApp')
    plot_historical('flesch_kincaid_grade', google_sub, boeing_sub, whatsapp_sub, 'Historical Development of Privacy Policy Readability', 'Grade Level', 'Google', 'Boeing', 'WhatsApp')


def add_statistics(descriptives, txt_file, df):
    for statistic in descriptives:
        txt_file.append(statistic)
        txt_file.append(df[statistic].mean())
        txt_file.append(df[statistic].median())
        txt_file.append(df[statistic].max())
        txt_file.append(df[statistic].min())
        txt_file.append(df[statistic].var())
        txt_file.append(df[statistic].std())
    return txt_file

#Function to determine Pearson R correlation from pandas dataframe
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

def plot_barplot(statistic, title, xlabel, ylabel, df):
    plt.figure()
    overall_median = df[statistic].median()
    dfg = df.groupby(['industry'])[statistic].median()
    ax = dfg.plot(kind='bar', title=title, color="olive", figsize=(8, 6))
    ax.set_title(title, fontsize=14, color="#222222")
    ax.set_ylabel(ylabel, fontsize=12, labelpad=12.0, color="#222222")
    ax.set_xlabel(xlabel, fontsize=12, labelpad=12.0, color="#222222")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=9, rotation=45, rotation_mode="anchor", ha="right", color="#222222")
    ax.axhline(y=overall_median, color="#222222", linestyle="--", lw=1)
    plt.tight_layout()
    plt.savefig("/Users/fabiankoenig/OneDrive - Harvard University/College/CS105/3_Your rights at Harvard/pp_txtfiles/output/plots/barplot_" + str(statistic) + ".png")
    plt.show()

def plot_boxplot(statistic, title, xlabel, df):
    c="olive"
    plt.figure(figsize=(6, 2))
    plt.boxplot(df[statistic],vert=0, labels=[""], patch_artist=True, medianprops=dict(color=c),
                boxprops=dict(facecolor="white", color="black"))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig("/Users/fabiankoenig/OneDrive - Harvard University/College/CS105/3_Your rights at Harvard/pp_txtfiles/output/plots/boxplot_" + str(statistic) + ".png")
    plt.show()

def plot_historical(statistic, company1_df, company2_df, company3_df, title, ylabel, company1, company2, company3):
    plt.figure()
    plt.plot(company1_df['date'], company1_df[statistic], label = company1)
    plt.plot(company2_df['date'], company2_df[statistic], label = company2)
    plt.plot(company3_df['date'], company3_df[statistic], label = company3)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig("/Users/fabiankoenig/OneDrive - Harvard University/College/CS105/3_Your rights at Harvard/pp_txtfiles/output/plots/historical_" + str(statistic) + ".png")
    plt.show()

def plot_scatter(xstatistic, ystatistic, title, xlabel, ylabel, df):
    plt.figure()
    fig, ax = plt.subplots()
    colors = {'Agriculture':'firebrick', 'Mining':'gold', 'Utilities':'darkgreen', 'Construction':'lightgreen', 'Manufacturing':'peru',
              'Wholesale Trade':'skyblue', 'Retail Trade':'darkblue', 'Transportation':'mediumseagreen', 'Information':'darkorange',
              'Finance and Insurance':'blueviolet', 'Real Estate':'violet', 'Professional, Scientific, and Technical Services':'cyan',
              'Management of Companies and Enterprises':'darkgray', 'Educational Services':'teal','Health Care':'yellowgreen',
              'Arts, Entertainment, and Recreation':'lightseagreen','Accommodation and Food Services':'crimson',}
    ax.scatter(df[xstatistic], df[ystatistic], c=df['industry'].map(colors), s=50)
    ax.set_ylim(ymin=0, ymax=90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=14, color="#222222")
    ax.set_ylabel(ylabel, fontsize=12, labelpad=12.0, color="#222222")
    ax.set_xlabel(xlabel, fontsize=12, labelpad=12.0, color="#222222")
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colors.values()]
    plt.legend(markers, colors.keys(), numpoints=1, ncol=2, loc="upper center", fontsize="x-small")
    plt.tight_layout()
    plt.savefig("/Users/fabiankoenig/OneDrive - Harvard University/College/CS105/3_Your rights at Harvard/pp_txtfiles/output/plots/scatter_" + str(ystatistic) + ".png")
    plt.show()

def plot_scatter_onlyindustry(xstatistic, ystatistic, title, xlabel, ylabel, df):
    plt.figure()
    fig, ax = plt.subplots()
    colors = {'Agriculture':'firebrick', 'Mining':'gold', 'Utilities':'darkgreen', 'Construction':'lightgreen', 'Manufacturing':'peru',
              'Wholesale Trade':'skyblue', 'Retail Trade':'darkblue', 'Transportation':'mediumseagreen', 'Information':'darkorange',
              'Finance and Insurance':'blueviolet', 'Real Estate':'violet', 'Professional, Scientific, and Technical Services':'cyan',
              'Management of Companies and Enterprises':'darkgray', 'Educational Services':'teal','Health Care':'yellowgreen',
              'Arts, Entertainment, and Recreation':'lightseagreen','Accommodation and Food Services':'crimson',}

    dfg_readingease = df.groupby(['industry'])[ystatistic].median()
    dfg_readingtime = df.groupby(['industry'])[xstatistic].median()

    ax.scatter(dfg_readingtime, dfg_readingease, c=dfg_readingtime.index.map(colors), s=50)
    ax.set_ylim(ymin=0, ymax=90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=14, color="#222222")
    ax.set_ylabel(ylabel, fontsize=12, labelpad=12.0, color="#222222")
    ax.set_xlabel(xlabel, fontsize=12, labelpad=12.0, color="#222222")
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in colors.values()]
    plt.legend(markers, colors.keys(), numpoints=1, ncol=2, loc="upper center", fontsize="x-small")
    plt.tight_layout()
    plt.savefig("/Users/fabiankoenig/OneDrive - Harvard University/College/CS105/3_Your rights at Harvard/pp_txtfiles/output/plots/scatterindustry_" + str(ystatistic) + ".png")
    plt.show()

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())

def replace_industry(naics, name, df):
    df['industry'] = df['industry'].replace([naics], str(name))

main()