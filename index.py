import numpy as np

# <***Extracting the datasets***>
sg_school_life_expectancy = np.genfromtxt('./Datasets/School Life Expectancy/school-life-expectancy-annual.csv', delimiter=',', names=True, dtype=['i8', 'f8'])
inter_school_life_expectancy = np.genfromtxt('./Datasets/School Life Expectancy/expected-years-of-schooling.csv', delimiter=',', names="Entity,Code,Year,Expected_Years_of_Schooling",  dtype=['U30', 'U4', 'i8', 'f8'], skip_header=1)
inter_mean_income = np.genfromtxt('./Datasets/Income/daily-mean-income.csv', delimiter=',', names=True,  dtype=['U30', 'U4', 'i8', 'f8'])
inter_unemployment_rate = np.genfromtxt('./Datasets/Unemployment Rates/unemployment-rate.csv', delimiter=',', names="Entity,Code,Year,Unemployment - total (% of total labor force) (modeled ILO estimate)",  dtype=['U30', 'U4', 'i8', 'f8'], skip_header=1)
inter_poverty_rate = np.genfromtxt('./Datasets/Poverty Rates/relative-poverty-share-of-people-below-60-of-the-median.csv', delimiter=',', names=True,  dtype=['U30', 'U4', 'i8', 'f8'])


'''# <***Cleaning the datasets***>
import csv
lines = list()
remove_values = ['Arab States (UNDP)', 'East Asia and the Pacific (UNDP)', 'Europe and Central Asia (UNDP)',
                 'High human development (UNDP)', 'Latin America and the Caribbean (UNDP)', 'Low human development (UNDP)',
                 'Medium human development (UNDP)', 'South Asia (UNDP)', 'Sub-Saharan Africa (UNDP)',
                 'Very high human development (UNDP)', 'World']
with open('./Datasets/School Life Expectancy/expected-years-of-schooling.csv', 'r') as read_file:
    reader = csv.reader(read_file)
    for row in enumerate(reader):
        if (row[1][0] not in remove_values):
            lines.append(row)
x = []
for i in range(len(lines)):
    x.append(lines[i][1])
#inter_school_life_expectancy = np.array(x)
del(x[0])
with open('./Datasets/School Life Expectancy/expected-years-of-schooling.csv', 'w') as write_file:
    writer = csv.writer(write_file)
    writer.writerows(x)'''


# <***Displaying the datasets***>
# School life expectancy in Sg
print(f'***School Life Expectancy in Singapore(from year 2000 to 2019)***')
print(f'\nThere are {len(sg_school_life_expectancy)} rows and {len(sg_school_life_expectancy.dtype)} columns in this dataset')
print(f'\nThe names of the columns are:')
for i in range(len(sg_school_life_expectancy.dtype)):
    print(f"- {sg_school_life_expectancy.dtype.names[i]} {type(sg_school_life_expectancy.dtype.names[i])} isnumeric: {str(sg_school_life_expectancy[1][i]).isnumeric()}")
print('\n')
for i in range(len(sg_school_life_expectancy.dtype)):
    unique_values = np.unique(sg_school_life_expectancy[sg_school_life_expectancy.dtype.names[i]])
    print(f'{len(unique_values)} unique values in {sg_school_life_expectancy.dtype.names[i]} column')

# School life expectancy internationally
print(f'\n***School Life Expectancy internationally***')
print(f'\nThere are {len(inter_school_life_expectancy)} rows and {len(inter_school_life_expectancy.dtype)} columns in this dataset')
print(f'\nThe names of the columns are:')
for i in range(len(inter_school_life_expectancy.dtype)):
    print(f"- {inter_school_life_expectancy.dtype.names[i]} {type(inter_school_life_expectancy.dtype.names[i])} isnumeric: {str(inter_school_life_expectancy[1][i]).isnumeric()}")
print('\n')
for i in range(len(inter_school_life_expectancy.dtype)):
    unique_values = np.unique(inter_school_life_expectancy[inter_school_life_expectancy.dtype.names[i]])
    print(f'{len(unique_values)} unique values in {inter_school_life_expectancy.dtype.names[i]} column')

# Mean daily income internationally
print(f'\n***Mean daily income internationally***')
print(f'\nThere are {len(inter_mean_income)} rows and {len(inter_mean_income.dtype)} columns in this dataset')
print(f'\nThe names of the columns are:')
for i in range(len(inter_mean_income.dtype)):
    print(f"- {inter_mean_income.dtype.names[i]} {type(inter_mean_income.dtype.names[i])} isnumeric: {str(inter_mean_income[1][i]).isnumeric()}")
print('\n')
for i in range(len(inter_mean_income.dtype)):
    unique_values = np.unique(inter_mean_income[inter_mean_income.dtype.names[i]])
    print(f'{len(unique_values)} unique values in {inter_mean_income.dtype.names[i]} column')

# Unemployment rate internationally
print(f'\n***Unemployment rate internationally***')
print(f'\nThere are {len(inter_unemployment_rate)} rows and {len(inter_unemployment_rate.dtype)} columns in this dataset')
print(f'\nThe names of the columns are:')
for i in range(len(inter_unemployment_rate.dtype)):
    print(f"- {inter_unemployment_rate.dtype.names[i]} {type(inter_unemployment_rate.dtype.names[i])} isnumeric: {str(inter_unemployment_rate[1][i]).isnumeric()}")
print('\n')
for i in range(len(inter_unemployment_rate.dtype)):
    unique_values = np.unique(inter_unemployment_rate[inter_unemployment_rate.dtype.names[i]])
    print(f'{len(unique_values)} unique values in {inter_unemployment_rate.dtype.names[i]} column')

# Poverty rate internationally
print(f'\n***Poverty rate internationally***')
print(f'\nThere are {len(inter_poverty_rate)} rows and {len(inter_poverty_rate.dtype)} columns in this dataset')
print(f'\nThe names of the columns are:')
for i in range(len(inter_poverty_rate.dtype)):
    print(f"- {inter_poverty_rate.dtype.names[i]} {type(inter_poverty_rate.dtype.names[i])} isnumeric: {str(inter_poverty_rate[1][i]).isnumeric()}")
print('\n')
for i in range(len(inter_poverty_rate.dtype)):
    unique_values = np.unique(inter_poverty_rate[inter_poverty_rate.dtype.names[i]])
    print(f'{len(unique_values)} unique values in {inter_poverty_rate.dtype.names[i]} column')


# <***Extracting Information from the datasets***>
# Extracting data values that are only needed (year: 2000-2019), (country: Singapore, USA, Indonesia)
# School life expectancy in USA and Indonesia
inter_school_life_expectancy_USA = inter_school_life_expectancy[inter_school_life_expectancy['Entity'] == 'United States' ]
inter_school_life_expectancy_Indonesia = inter_school_life_expectancy[inter_school_life_expectancy['Entity'] == 'Indonesia' ]
# 2000-2019
inter_school_life_expectancy_USA_year = inter_school_life_expectancy_USA[10: 30]
inter_school_life_expectancy_Indonesia_year = inter_school_life_expectancy_Indonesia[10: 30]


# Mean daily income in USA and Indonesia
inter_mean_income_USA = inter_mean_income[inter_mean_income['Entity'] == 'United States']
inter_mean_income_Indonesia = inter_mean_income[inter_mean_income['Entity'] == 'Indonesia']
# 2000-2019
inter_mean_income_USA_year = inter_mean_income_USA[12: 32]
inter_mean_income_Indonesia_year = inter_mean_income_Indonesia[7: 27]

# Unemployment rate in Singapore, USA and Indonesia
inter_unemployment_rate_Singapore = inter_unemployment_rate[inter_unemployment_rate['Entity'] == 'Singapore']
inter_unemployment_rate_USA = inter_unemployment_rate[inter_unemployment_rate['Entity'] == 'United States']
inter_unemployment_rate_Indonesia = inter_unemployment_rate[inter_unemployment_rate['Entity'] == 'Indonesia']
# 2000-2019
inter_unemployment_rate_Singapore_year = inter_unemployment_rate_Singapore[9: 29]
inter_unemployment_rate_USA_year = inter_unemployment_rate_USA[9: 29]
inter_unemployment_rate_Indonesia_year = inter_unemployment_rate_Indonesia[9: 29]

# Poverty rate in USA and Indonesia
inter_poverty_rate_USA = inter_poverty_rate[inter_poverty_rate['Entity'] == 'United States']
inter_poverty_rate_Indonesia = inter_poverty_rate[inter_poverty_rate['Entity'] == 'Indonesia']
# 2000-2019
inter_poverty_rate_USA_year = inter_poverty_rate_USA[12: 32]
inter_poverty_rate_Indonesia_year = inter_poverty_rate_Indonesia[7: 27]


# <***Plotting the Charts***>
import matplotlib.pyplot as plt


# **Overview of Data**
plt.figure(1, figsize=(16, 9))
# Histogram of School life expectancy means of each country
plt.subplot(121)
school_life_expectancy_each_country = []
no_of_country = np.unique(inter_school_life_expectancy["Entity"])
for i in range(len(no_of_country)):
    sum = 0
    x = 0
    for j in range(len(inter_school_life_expectancy)):
        if no_of_country[i] == inter_school_life_expectancy["Entity"][j] and inter_school_life_expectancy["Year"][j] > 1999 and inter_school_life_expectancy["Year"][j] <2020:
            sum += inter_school_life_expectancy["Expected_Years_of_Schooling"][j]
            x += 1
    mean = sum/x
    school_life_expectancy_each_country.append(mean)

plt.hist(school_life_expectancy_each_country, color="orange")
plt.title("Distribution of school life expectancy mean of each country")
plt.xlabel("Frequency")
plt.ylabel("School Life Expectancy(years)")

# Line graph of School life expectancy means of all countries over the years
plt.subplot(122)
school_life_expectancy_each_year = []
year = 2000
while year < 2020:
    for i in range(20):
        sum = 0
        x = 0
        for j in range(len(inter_school_life_expectancy)):
            if inter_school_life_expectancy["Year"][j] == year:
                sum += inter_school_life_expectancy["Expected_Years_of_Schooling"][j]
                x += 1
        mean = sum/x
        school_life_expectancy_each_year.append(mean)
        year += 1

years = np.arange(2000, 2020)
plt.plot(years, school_life_expectancy_each_year)
plt.title("Mean World School life expectancy over time")
plt.xlabel("Years")
plt.ylabel("School Life Expectancy(years)")


# **Data for each country**
# *Singapore*
plt.figure(2, figsize=(16, 9))
# Line graph of School life expectancy over the years for Singapore
plt.subplot(221)
years = sg_school_life_expectancy["year"]
school_life_expectancy = sg_school_life_expectancy["school_life_expectancy"]
plt.plot(years, school_life_expectancy)
plt.title("School life expectancy over time for Singapore")
plt.xlabel("Years")
plt.ylabel("School Life Expectancy(years)")

# Line graph of the rate of unemployment in Singapore
plt.subplot(222)
years = inter_unemployment_rate_Singapore_year["Year"]
unemployment = inter_unemployment_rate_Singapore_year["Unemployment__total__of_total_labor_force_modeled_ILO_estimate"]
plt.plot(years, unemployment)
plt.title("Rate of unemployment over time for Singapore")
plt.xlabel("Years")
plt.ylabel("Rate of unemployment")

# Pie chart of the rate of unemployment in Singapore
plt.subplot(223)
inter_unemployment_rate_Singapore_year_mean = np.array([np.mean(inter_unemployment_rate_Singapore_year["Unemployment__total__of_total_labor_force_modeled_ILO_estimate"])])
non_unemployment_rate_Singapore = np.array([100]) - inter_unemployment_rate_Singapore_year_mean
values = np.concatenate((inter_unemployment_rate_Singapore_year_mean, non_unemployment_rate_Singapore))
labels = ['People unemployed', 'People employed']
myexplode = [0.2, 0]
colors = ['lightcoral', 'paleturquoise']
plt.pie(values, labels=labels, explode=myexplode, shadow=True, autopct='%1.1f%%', colors=colors)
plt.title("Rate of unemployment in Singapore")

# *USA*
plt.figure(3, figsize=(16, 9))
# Line graph of School life expectancy over the years for USA
plt.subplot(221)
years = inter_school_life_expectancy_USA_year["Year"]
school_life_expectancy = inter_school_life_expectancy_USA_year["Expected_Years_of_Schooling"]
plt.plot(years, school_life_expectancy)
plt.title("School life expectancy over time for USA")
plt.xlabel("Years")
plt.ylabel("School Life Expectancy(years)")

# Line graph of daily mean income over the years for USA
plt.subplot(222)
years = inter_mean_income_USA_year["Year"]
daily_mean_income = inter_mean_income_USA_year["Mean_income_or_expenditure_per_day"]
plt.plot(years, daily_mean_income)
plt.title("Daily mean income over time for USA")
plt.xlabel("Years")
plt.ylabel("Daily Mean Income(USD)")

# Pie chart of Poverty Rates in USA
plt.subplot(223)
inter_poverty_rate_USA_year_mean = np.array([np.mean(inter_poverty_rate_USA_year["60_of_median__share_of_population_below_poverty_line"])])
non_poverty_rate_USA = np.array([100]) - inter_poverty_rate_USA_year_mean
values = np.concatenate((inter_poverty_rate_USA_year_mean, non_poverty_rate_USA))
labels = ['People living in poverty', 'People not living in poverty']
myexplode = [0.2, 0]
colors = ['sandybrown', 'powderblue']
plt.pie(values, labels=labels, explode=myexplode, shadow=True, autopct='%1.1f%%', colors=colors)
plt.title("Rate of poverty in USA")

# Pie chart of the rate of unemployment in USA
plt.subplot(224)
inter_unemployment_rate_USA_year_mean = np.array([np.mean(inter_unemployment_rate_USA_year["Unemployment__total__of_total_labor_force_modeled_ILO_estimate"])])
non_unemployment_rate_USA = np.array([100]) - inter_unemployment_rate_USA_year_mean
values = np.concatenate((inter_unemployment_rate_USA_year_mean, non_unemployment_rate_USA))
labels = ['People unemployed', 'People employed']
myexplode = [0.2, 0]
colors = ['lightcoral', 'paleturquoise']
plt.pie(values, labels=labels, explode=myexplode, shadow=True, autopct='%1.1f%%', colors=colors)
plt.title("Rate of unemployment in USA")

# *Indonesia*
plt.figure(4, figsize=(16, 9))
# Line graph of School life expectancy over the years for Indonesia
plt.subplot(221)
years = inter_school_life_expectancy_Indonesia_year["Year"]
school_life_expectancy = inter_school_life_expectancy_Indonesia_year["Expected_Years_of_Schooling"]
plt.plot(years, school_life_expectancy)
plt.title("School life expectancy over time for Indonesia")
plt.xlabel("Years")
plt.ylabel("School Life Expectancy(years)")

# Line graph of daily mean income over the years for Indonesia
plt.subplot(222)
years = inter_mean_income_Indonesia_year["Year"]
daily_mean_income = inter_mean_income_Indonesia_year["Mean_income_or_expenditure_per_day"]
plt.plot(years, daily_mean_income)
plt.title("Daily mean income over time for Indonesia")
plt.xlabel("Years")
plt.ylabel("Daily Mean Income(USD)")

# Pie chart of Poverty Rates in Indonesia
plt.subplot(223)
inter_poverty_rate_Indonesia_year_mean = np.array([np.mean(inter_poverty_rate_Indonesia_year["60_of_median__share_of_population_below_poverty_line"])])
non_poverty_rate_Indonesia = np.array([100]) - inter_poverty_rate_Indonesia_year_mean
values = np.concatenate((inter_poverty_rate_Indonesia_year_mean, non_poverty_rate_Indonesia))
labels = ['People living in poverty', 'People not living in poverty']
myexplode = [0.2, 0]
colors = ['sandybrown', 'powderblue']
plt.pie(values, labels=labels, explode=myexplode, shadow=True, autopct='%1.1f%%', colors=colors)
plt.title("Rate of poverty in Indonesia")

# Pie chart of the rate of unemployment in Indonesia
plt.subplot(224)
inter_unemployment_rate_Indonesia_year_mean = np.array([np.mean(inter_unemployment_rate_Indonesia_year["Unemployment__total__of_total_labor_force_modeled_ILO_estimate"])])
non_unemployment_rate_Indonesia = np.array([100]) - inter_unemployment_rate_Indonesia_year_mean
values = np.concatenate((inter_unemployment_rate_Indonesia_year_mean, non_unemployment_rate_Indonesia))
myexplode = [0.2, 0]
colors = ['lightcoral', 'paleturquoise']
labels = ['People unemployed', 'People employed']
plt.pie(values, labels=labels, explode=myexplode, shadow=True, autopct='%1.1f%%', colors=colors)
plt.title("Rate of unemployment in Indonesia")


# **Comparing all the factors for each country**
fig = plt.figure(5, figsize=(16, 9))
# *USA*
# Scatter plot of School life expectancy by income, poverty and unemployment in the USA
ax = fig.add_subplot(121)
school_life_expectancy = inter_school_life_expectancy_USA_year["Expected_Years_of_Schooling"]
daily_mean_income = inter_mean_income_USA_year["Mean_income_or_expenditure_per_day"]
poverty_rate = inter_poverty_rate_USA_year["60_of_median__share_of_population_below_poverty_line"]
plt.scatter(school_life_expectancy, daily_mean_income, linewidths=1, alpha=.7, edgecolor='k', s=200, c=poverty_rate)
plt.title("Scatter plot of School life expectancy by income and poverty in the USA")
m,b = np.polyfit(school_life_expectancy, daily_mean_income, deg=1)
plt.plot(school_life_expectancy, m*school_life_expectancy + b, 'r-')
plt.xlabel("School Life Expectancy(years)")
plt.ylabel("Mean Daily Income(USD)")
ax.text(15.5, 88.5, 'The darker the color(yellow(light) to purple(dark)), '
                'the higher the rate of poverty', style='italic',
                bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

# *Indonesia*
# Scatter plot of School life expectancy by income, poverty and unemployment in Indonesia
ax = fig.add_subplot(122)
school_life_expectancy = inter_school_life_expectancy_Indonesia_year["Expected_Years_of_Schooling"]
daily_mean_income = inter_mean_income_Indonesia_year["Mean_income_or_expenditure_per_day"]
poverty_rate = inter_poverty_rate_Indonesia_year["60_of_median__share_of_population_below_poverty_line"]
plt.scatter(school_life_expectancy, daily_mean_income, linewidths=1, alpha=.7, edgecolor='k', s=200, c=poverty_rate)
plt.title("Scatter plot of School life expectancy by income and poverty in Indonesia")
m,b = np.polyfit(school_life_expectancy, daily_mean_income, deg=1)
plt.plot(school_life_expectancy, m*school_life_expectancy + b, 'r-')
plt.xlabel("School Life Expectancy(years)")
plt.ylabel("Mean Daily Income(USD)")


# **Comparing between the three countries**
plt.figure(6, figsize=(16, 9))
# Bar Charts comparing the three countries in terms of school life expectancy, income, poverty and unemployment
# Bar Chart School Life Expectancy
plt.subplot(221)
countries = np.arange(3)
singapore_mean_SLE = np.array([np.mean(sg_school_life_expectancy["school_life_expectancy"])])
usa_mean_SLE = np.array([np.mean(inter_school_life_expectancy_USA_year["Expected_Years_of_Schooling"])])
indonesia_mean_SLE = np.array([np.mean(inter_school_life_expectancy_Indonesia_year["Expected_Years_of_Schooling"])])
school_life_expectancies = np.concatenate((singapore_mean_SLE, usa_mean_SLE, indonesia_mean_SLE))

plt.bar(countries, school_life_expectancies, color='darkviolet')
plt.ylabel("Mean School Life Expectancy(years)")
plt.title("Mean School Life Expectancies of the three countries")
plt.xticks(countries, ('Singapore', 'USA', 'Indonesia'))

# Bar Chart Daily Mean Income
plt.subplot(222)
countries = np.arange(3)
usa_daily_mean_income = np.array([np.mean(inter_mean_income_USA_year["Mean_income_or_expenditure_per_day"])])
indonesia_daily_mean_income = np.array([np.mean(inter_mean_income_Indonesia_year["Mean_income_or_expenditure_per_day"])])
daily_mean_incomes = np.concatenate((np.array([0]), usa_daily_mean_income, indonesia_daily_mean_income))

plt.bar(countries, daily_mean_incomes, color='springgreen')
plt.ylabel("Mean Daily Income(USD)")
plt.title("Mean Daily Income of the three countries")
plt.xticks(countries, ('Singapore', 'USA', 'Indonesia'))

# Bar Chart Poverty and Unemployment
plt.subplot(223)
countries = np.arange(3)

usa_poverty_rate = np.array([np.mean(inter_poverty_rate_USA_year["60_of_median__share_of_population_below_poverty_line"])])
indonesia_poverty_rate = np.array([np.mean(inter_poverty_rate_Indonesia_year["60_of_median__share_of_population_below_poverty_line"])])
y1_poverty_rate = np.concatenate((np.array([0]), usa_poverty_rate, indonesia_poverty_rate))

singapore_unemployment_rate = np.array([np.mean(inter_unemployment_rate_Singapore_year["Unemployment__total__of_total_labor_force_modeled_ILO_estimate"])])
usa_unemployment_rate = np.array([np.mean(inter_unemployment_rate_USA_year["Unemployment__total__of_total_labor_force_modeled_ILO_estimate"])])
indonesia_unemployment_rate = np.array([np.mean(inter_unemployment_rate_Indonesia_year["Unemployment__total__of_total_labor_force_modeled_ILO_estimate"])])
y2_unemployment_rate = np.concatenate((singapore_unemployment_rate, usa_unemployment_rate, indonesia_unemployment_rate))

width = 0.4
plt.bar(countries-0.2, y1_poverty_rate, width, color='cyan')
plt.bar(countries+0.2, y2_unemployment_rate, width, color='orange')
plt.title("Percentage/Rate of Poverty and Unemployment in the three countries")
plt.xticks(countries, ('Singapore', 'USA', 'Indonesia'))
plt.xlabel("Countries")
plt.ylabel("Percentage/Rate")
plt.legend(["Poverty Rate", "Unemployment Rate"])

plt.show()