
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from datetime import datetime
import os
import re
import unittest


#task1
class FileOperation:

    def read_excel(self, file_path: str):
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"An error occurred while reading the Excel file: {e}")
            return None

    def save_to_excel(self, data, file_name: str):
        try:
            data.to_excel(file_name, index=False)
            print(f"Data successfully saved to {file_name}")
        except Exception as e:
            print(f"An error occurred while saving the data to Excel: {e}")

#task2
class SalesData:
    def __init__(self, data):
        self.data = data

    def eliminate_duplicates(self):
        # מסנן את השורות הכפולות
        self.data.drop_duplicates(inplace=True)
        # מסנן את השורות עם ערכים חסרים
        self.data.dropna(inplace=True)

    def calculate_total_sales(self):
        # חישוב סכום המכירות עבור כל מוצר
        total_sales_per_product = self.data.groupby('Product')['Quantity'].sum()
        return total_sales_per_product

    def _calculate_total_sales_per_month(self):
        # חישוב סכום המכירות עבור כל חודש
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d.%m.%Y')
        self.data['Month'] = self.data['Date'].dt.month
        total_sales_per_month = self.data.groupby('Month')['Total'].sum()
        return total_sales_per_month

    def _identify_best_selling_product(self):
        # זיהוי המוצר הנמכר ביותר
        best_selling_product = self.data.groupby('Product')['Quantity'].sum().idxmax()
        return best_selling_product

    def _identify_month_with_highest_sales(self):
        # זיהוי החודש עם המכירות הגבוהות ביותר
        total_sales_per_month = self._calculate_total_sales_per_month()
        month_with_highest_sales = total_sales_per_month.idxmax()
        return month_with_highest_sales

    def analyze_sales_data(self):
        # ניתוח הנתונים על פי המתואר בדרישות
        best_selling_product = self._identify_best_selling_product()
        month_with_highest_sales = self._identify_month_with_highest_sales()
        analysis_results = {
            'best_selling_product': best_selling_product,
            'month_with_highest_sales': month_with_highest_sales
        }
        return analysis_results

    #task3

    def calculate_cumulative_sales(self):
      # קביעת המחיר של כל מוצר כעמודת נפרדת
        self.data['Total_Price'] = self.data['Price'] * self.data['Quantity']

      # חישוב סכום המכירות הצטבריות לכל מוצר
        cumulative_sales = self.data.groupby('Product')['Total_Price'].cumsum()

        return cumulative_sales

    def add_90_percent_values_column(self):
        # חישוב הערך של 90% מהעמודה 'Quantity'
        ninety_percent_values = self.data['Quantity'].quantile(0.9)

        # הוספת עמודה חדשה למסגרת הנתונים עם הערכים המחושבים
        self.data['90%_Values'] = ninety_percent_values

        return self.data

    def calculate_mean_quantity(self):
        # קבלת הנתונים מעמודת Total
        total_values = self.data['Total'].values

        # חישוב הממוצע, החציון והערך השני הגדול
        mean = np.mean(total_values)
        median = np.median(total_values)
        second_max = np.partition(total_values, -2)[-2]

        return mean, median, second_max

    def bar_chart_category_sum(self):
        # קבוצת הנתונים לפי מוצר וחישוב סכום כמויות המכירות לכל מוצר
        category_sum = self.data.groupby('Product')['Quantity'].sum().reset_index()

        # ציור התרשים
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Product', y='Quantity', data=category_sum)
        plt.title('Sum of Quantities Sold for Each Product')
        plt.xlabel('Product')
        plt.ylabel('Sum of Quantity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def filter_by_sellings_or_and(self):
            # סינון המוצרים לפי התנאים הנתונים
            filtered_data = self.data[(self.data['Quantity'] > 5) | (self.data['Quantity'] == 0)]
            filtered_data = filtered_data[(filtered_data['Price'] > 300) & (filtered_data['Quantity'] < 2)]

            return filtered_data

    def add_black_friday_price_column(self):
     # Add a new column named 'BlackFridayPrice' to the SalesData DataFrame
     self.data['BlackFridayPrice'] = self.data['Price'] / 2

    def calculate_stats(self, columns: str = None):
         if columns is None:
             # If columns is None, calculate stats for all columns
             columns = self.data.columns

         stats_dict = {}
         for column in columns:
             # Calculate stats for each column
             max_val = self.data[column].max()
             sum_val = self.data[column].sum()
             abs_val = self.data[column].abs()
             cummax_val = self.data[column].cummax()

             # Create a dictionary to store the stats for the current column
             column_stats = {
                 'max': max_val,
                 'sum': sum_val,
                 'absolute': abs_val,
                 'cumulative_max': cummax_val
             }

             # Add the column stats dictionary to the stats_dict
             stats_dict[column] = column_stats

         return stats_dict

    #task4-רשות
    #task5-רשות

    #task 6

    def seaborn_plot_1(self):
        # הוספת שרטוט סטריפ
        sns.stripplot(x='Product', y='Quantity', data=self.data)
        plt.title('Strip Plot of Quantity per Product')
        plt.xlabel('Product')
        plt.ylabel('Quantity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def seaborn_plot_2(self):
        # הוספת שרטוט סקטר
        sns.scatterplot(x='Price', y='Quantity', hue='Product', data=self.data)
        plt.title('Scatter Plot of Quantity vs. Price')
        plt.xlabel('Price')
        plt.ylabel('Quantity')
        plt.tight_layout()
        plt.show()

    def seaborn_plot_3(self):
        # הוספת שרטוט תרשים פירמידה
        sns.barplot(x='Product', y='Total', data=self.data)
        plt.title('Bar Plot of Total Sales per Product')
        plt.xlabel('Product')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def seaborn_plot_4(self):
            sns.pairplot(self.data)
            plt.title('Pair Plot')
            plt.show()

    def seaborn_plot_5(self):
        # הוספת שרטוט קונט
        sns.countplot(x='Month', data=self.data)
        plt.title('Count Plot of Sales per Month')
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    def matplotlib_plot_1(self):
        # הוספת שרטוט קו
        plt.plot(self.data['Date'], self.data['Quantity'])
        plt.title('Line Plot of Quantity over Time')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def matplotlib_plot_2(self):
        # הוספת שרטוט עמודות
        plt.bar(self.data['Product'], self.data['Quantity'])
        plt.title('Bar Plot of Quantity per Product')
        plt.xlabel('Product')
        plt.ylabel('Quantity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def matplotlib_plot_3(self):
        # הוספת שרטוט פיצ'ה
        plt.pie(self.data.groupby('Product')['Quantity'].sum(), labels=self.data['Product'].unique(), autopct='%1.1f%%')
        plt.title('Pie Chart of Quantity per Product')
        plt.tight_layout()
        plt.show()

    def matplotlib_plot_4(self):
        # הוספת שרטוט פיצ'ה עם נתוני שוליים
        explode = (0.1, 0, 0, 0, 0)  # הפרדה של קטעים
        plt.pie(self.data.groupby('Product')['Quantity'].sum(), labels=self.data['Product'].unique(), autopct='%1.1f%%',
                explode=explode)
        plt.title('Exploded Pie Chart of Quantity per Product')
        plt.tight_layout()
        plt.show()

    def matplotlib_plot_5(self):
        # הוספת שרטוט אזור
        plt.fill_between(self.data['Date'], self.data['Quantity'], color='skyblue', alpha=0.4)
        plt.plot(self.data['Date'], self.data['Quantity'], color='Slateblue', alpha=0.6)
        plt.title('Area Plot of Quantity over Time')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def matplotlib_plot_6(self):
            plt.scatter(self.data['Price'], self.data['Total'], color='green', alpha=0.5)
            plt.title('Scatter Plot of Price vs. Total Sales')
            plt.xlabel('Price')
            plt.ylabel('Total Sales')
            plt.show()

    def matplotlib_plot_7(self):
        plt.hist(self.data['Price'], bins=20, color='skyblue', edgecolor='black')
        plt.title('Histogram of Price')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.show()

    #task 7

    def func1(self):
        try:
            # הוספת השגיאה כאן
            raise TypeError("value is invalid")
        except TypeError as e:
            print(f"<Sari & Giti, {datetime.now().strftime('%d.%m.%Y, %H:%M')}> {e} <Sari & Giti>")

    def read_file(file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        return data

    def random_sales_and_highest_amount(product, sales_range):
        random_sales = random.randint(*sales_range)
        random_amount = random.uniform(0, 1000)
        return random_sales, random_amount

    def func4(self):
        print(sys.version)

    def process_parameters(*args, **kwargs):
        result = {}
        for arg in args:
            if isinstance(arg, str) and arg.startswith('<') and arg.endswith('>'):
                key = arg.strip('<>')
                result[key] = kwargs.get(key)
            else:
                print(arg)
        return result

    def func6(self):
        # הדפסת שלושה שורות ראשונות
        print(data.head(3))

        # הדפסת שני שורות אחרונות
        print(data.tail(2))

        # הדפסת שורה אקראית
        print(data.sample(1))

    def func7(self):
        for column in data.select_dtypes(include=['number']).columns:
            for value in data[column]:
                print(value)

#task 8
def check_and_create_file(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'w') as file:
            file.write('')
    else:
        print(f"File {file_path} exists.")

def read_usernames(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

def read_and_validate_emails(file_path):
    with open(file_path, 'r') as file:
        emails = [line.strip() for line in file]
        valid_emails = [email for email in emails if validate_email(email)]
        return valid_emails

def validate_email(email):
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return True
    else:
        return False

def filter_gmail_addresses(emails):
    gmail_addresses = [email for email in emails if email.endswith('@gmail.com')]
    return gmail_addresses

def read_usernames_to_array(file_path):
    with open(file_path, 'r') as file:
        usernames = [line.strip() for line in file]
        ten_percent = len(usernames) // 10
        usernames_subset = usernames[:ten_percent]
        return usernames_subset

def read_even_users(file_path):
    with open(file_path, 'r') as file:
        even_users = [line.strip() for index, line in enumerate(file) if index % 2 == 0]
        return even_users

def match_usernames_to_emails(usernames, emails):
    for username, email in zip(usernames, emails):
        if username.lower() in email.lower():
            print(f"Username {username} matches email {email}.")
        else:
            print(f"No match found for username {username} and email {email}.")

def count_a_in_username(username):
    a_count = username.lower().count('a')
    return a_count

def check_username_presence(username, users_list):
    if username in users_list:
        username_ascii = ''.join(str(ord(char)) for char in username)
        username_str = ''.join(chr(int(username_ascii[i:i+2])) for i in range(0, len(username_ascii), 2))
        a_count = count_a_in_username(username_str)
        print(f"Username {username} found in the list. Count of 'A's in the name: {a_count}.")
    else:
        print(f"Username {username} not found in the list.")

def capitalize_usernames(usernames):
    capitalized_usernames = [username.upper() for username in usernames]
    return capitalized_usernames

def calculate_payment(customers):
    total_payment = 0
    for index, customer in enumerate(customers, start=1):
        if index % 8 == 0:
            total_payment += 200
        elif index % 8 == 1:
            total_payment += 250
    return total_payment

def main():
    check_and_create_file("UsersEmail.txt")
    check_and_create_file("UsersName.txt")
    check_and_create_file("YafeNof.csv")

    valid_emails = read_and_validate_emails("UsersEmail.txt")
    gmail_addresses = filter_gmail_addresses(valid_emails)

    usernames_array = read_usernames_to_array("UsersName.txt")
    even_users = read_even_users("UsersName.txt")
    match_usernames_to_emails(usernames_array, valid_emails)
    check_username_presence("Alice", usernames_array)
    capitalized_usernames = capitalize_usernames(usernames_array)

    payment = calculate_payment([9, 5, 19, 43, 4, 88, 76, 20, 15])
    print(f"Total payment for the group: {payment} NIS.")


import unittest

class TestSalesData(unittest.TestCase):
    def setUp(self):
        # יצירת אובייקט FileOperation כדי לקרוא את הנתונים מהקובץ
        file_op = FileOperation()
        # קריאה לפונקציה read_excel כדי לקרוא את הנתונים מהקובץ
        self.sales_data = file_op.read_excel("YafeNof.csv")

    def test_eliminate_duplicates(self):
        # בדיקה שהפונקציה מסננת כפילויות וערכים חסרים
        sales_obj = SalesData(self.sales_data)
        sales_obj.eliminate_duplicates()
        self.assertEqual(len(sales_obj.data), 80)  # מספר השורות ישר אחרי המסנים

    def test_calculate_total_sales(self):
        # בדיקה שהפונקציה מחשבת סכום מכירות נכון לכל מוצר
        sales_obj = SalesData(self.sales_data)
        total_sales_per_product = sales_obj.calculate_total_sales()
        self.assertEqual(total_sales_per_product['Sidur'], 150)  # סכום מכירות עבור סידור
        self.assertEqual(total_sales_per_product['Teilim'], 117)  # סכום מכירות עבור תהילים

    def test_calculate_total_sales_per_month(self):
        # בדיקה שהפונקציה מחשבת סכום מכירות לכל חודש
        sales_obj = SalesData(self.sales_data)
        total_sales_per_month = sales_obj._calculate_total_sales_per_month()
        self.assertEqual(total_sales_per_month[1], 80)  # סכום מכירות בינואר
        self.assertEqual(total_sales_per_month[2], 30)  # סכום מכירות בפברואר

    def test_identify_best_selling_product(self):
        # בדיקה שהפונקציה מזהה את המוצר הנמכר ביותר
        sales_obj = SalesData(self.sales_data)
        best_selling_product = sales_obj._identify_best_selling_product()
        self.assertEqual(best_selling_product, 'Tanach')  # המוצר הנמכר ביותר



# Press the green button in the gutter to run the script.
#if __name__ == '__main__':



    #נסיונות הרצה 

    #file_op = FileOperation()
    #yafe_nof = file_op.read_excel("YafeNof.csv")
#print(yafe_nof)
#file_op.save_to_excel(yafe_nof, "YafeNof_new.xlsx")
#sales_obj = SalesData(yafe_nof)

#sales_obj.eliminate_duplicates()
#print(yafe_nof)
#
# total_sales_per_product = sales_obj.calculate_total_sales()
# print("Total sales per product:")
# print(total_sales_per_product)

#total_sales_per_month = sales_obj._calculate_total_sales_per_month()
#print("\nTotal sales per month:")
#print(total_sales_per_month)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#month_with_highest_sales = sales_obj._identify_month_with_highest_sales()
#print("\nMonth with highest sales:")
#print(month_with_highest_sales)

# analysis_results = sales_obj.analyze_sales_data()
# print("\nAnalysis results:")
# print(analysis_results)

#-------------3------------

# יצירת אובייקט SalesData

# #11
# # הרצת פונקציה לחישוב סכום המכירות הצטבריות לכל מוצר לאורך החודשים
# cumulative_sales = sales_obj.calculate_cumulative_sales()
# print("Cumulative sales per product:")
# print(cumulative_sales)
# #12
# result = sales_obj.add_90_percent_values_column()
# print(result)
# #13
# #sales_obj.bar_chart_category_sum()
# #14
# print("----------14--------")
# mean, median, second_max = sales_obj.calculate_mean_quantity()
# print("Mean:", mean)
# print("Median:", median)
# print("Second Max:", second_max)
#15
#filtered_data = sales_obj.filter_by_sellings_or_and()
#print(filtered_data)
#16
#sales_obj.add_black_friday_price_column()
# print(sales_obj.data)
#17
# ייבא את המודולים שנדרשים


# יצירת מופע של מחלקת SalesData

# קריאה לפונקציה calculate_stats עבור עמודות ספציפיות או ללא ציון עמודות (במקרה זה, כל העמודות)
#stats = sales_obj.calculate_stats(columns=['Price', 'Quantity', 'Total'])

#sales_obj.eliminate_duplicates()
#sales_obj.calculate_total_sales()

# הרצת הפונקציות לשרטוט שרטוטים עם Seaborn ו- Matplotlib
#sales_obj.seaborn_plot_1()
#sales_obj.seaborn_plot_2()
#sales_obj.seaborn_plot_3()
#sales_obj.seaborn_plot_4()
#sales_obj.seaborn_plot_5()

#sales_obj.matplotlib_plot_1()
#sales_obj.matplotlib_plot_2()
#sales_obj.matplotlib_plot_3()
#sales_obj.matplotlib_plot_4()
#sales_obj.matplotlib_plot_5()
#sales_obj.matplotlib_plot_6()
#sales_obj.matplotlib_plot_7()
#main()





