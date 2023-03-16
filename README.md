# Proekt_taxi
# «Анализ данных крупного агрегатора такси (поиск инсайтов, проверка гипотезы, составление
# рекомендаций стейкхолдерам)».
# Исследовать данные, проверить гипотезу, сделать выводы презентовать результаты.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
# Загрузим файл для работы и посмотрим что за данные у нас имеются
df = pd.read_csv('dip_hw_x_taxi.csv')

# Посмотрим распределение значений по столбцам в таблице
df.describe()

# Проверяем чтобы все данные были заполнены и с дата фреймом можно работать
df.info()

df.value_counts(['offer_class_group'])

# Построим график распределения дистанции поездок
plt.hist(df['distance_km'], color = 'blueviolet', edgecolor = 'white', bins = int(2000/80))
plt.xlabel("Расстояние поездок, км.")
plt.ylabel("Количество поездок")

# Создаем  новый дата фрейм, оставляю только совершенные поездки
df_filter = df[df['driver_response'] == 1]

# Строим график распределения дистанции поездок без отмененных заказов
plt.hist(df_filter['distance_km'], color = 'blueviolet', edgecolor = 'white', bins = int(2000/80))
plt.xlabel("Расстояние поездок, км.")
plt.ylabel("Количество поездок")

# Для дальнейшего анализа использовались данные по совершенным поездкам, так как отмененные поездки дают большую погрешность в показателе частоты заказов.
# Заводим новые переменные для сокращения символов
Econ = df_filter[df_filter['offer_class_group'] == 'Economy']
Comf = df_filter[df_filter['offer_class_group'] == 'Comfort']
Deliv = df_filter[df_filter['offer_class_group'] == 'Delivery']
Prem = df_filter[df_filter['offer_class_group'] == 'Premium']

# Визуализируйте распределения дистанций поездок, по каждому классу такси в отдельности, а также определите типы распределений для
# эконом-класса и комфорт-класса

# Создадим списки разделив поездки по классам. Для того, чтобы потом построить график дистанции поездок по каждому классу.

Economy = list(Econ['distance_km'])
Comfort	= list(Comf['distance_km'])
Delivery = list(Deliv['distance_km'])
Premium	= list(Prem['distance_km'])

# Пропишем цвета и названия, которые для графика будут использованы
colors = ['deepskyblue', 'gold', 'lightgreen', 'fuchsia']
names = ['Economy', 'Comfort', 'Delivery', 'Premium']

# Строим график распределения расстояний поездок по классам такси
plt.hist([Economy, Comfort, Delivery, Premium], bins = int(2000/200), color = colors, label=names)
plt.legend()
plt.xlabel("Расстояние поездок, км.")
plt.ylabel("Количество поездок")

# Сформируйте выборки по эконом и комфорт-классам. Визуализируйте пересечение интервалов дистанций этих классов

# Строим график распределения расстояний поездок только для Эконома и Комфорта, чтобы удобнее было делать вывод о типе распределения.
colors_2 = ['deepskyblue', 'gold']
names_2 = ['Economy', 'Comfort']
plt.hist([Economy, Comfort], bins = int(2000/200), color = colors_2, label=names_2)
plt.legend()
plt.xlabel("Расстояние поездок, км.")
plt.ylabel("Количество поездок")

# Создаем новый датафрейм с поездками только эконом-классом и комфорт
df_selection = pd.concat([Econ,Comf])

# Строим график пересечений дистанций эконом-класса и комфорт. x - это параметр по которому строим график, hue = параметр, по которому будет разбивка по цветам.
# kind = тип построения графика, тк по умолчанию будет гистограмма.
sns.displot(data = df_selection, x = 'distance_km', hue = 'offer_class_group', kind='kde')
plt.show()

# Проведите стат. тест (поможет: функция scipy.stats.ttest_ind), проверьте гипоетезу, что дистанции поездок в комфорт-классе отличаются от
# дистанций поездок эконом-класса (примечание: H0 – выборки не отличаются, H1 – выборки отличны; уровень значимости = 5%)

# Создаем массив по эконому для дальнейшего t-теста
economy_array = list(Econ['distance_km'])

# Создаем массив по комфорту для дальнейшего t-теста
comfort_array = list(Comf['distance_km'])

# Делаю другой список, чтобы при анализе выборки были одного размера
comfort_array_2 = comfort_array[0:211]

#  Проверяю длину выборок, должны быть одной длины
len(economy_array), len(comfort_array_2)

# Для работы сравниваем дисперсии каждой совокупности, чтобы быть уверенными что дисперсии одинаковые (должно быть соотношение большей к меньшей не больше 4:1)
print(np.var(economy_array), np.var(comfort_array_2))

# Выполняю двухвыборочный t-критерий с равными дисперсиями
stats.ttest_ind(a=comfort_array_2, b=economy_array, equal_var=True)

# Проверяем, не превышает ли вычисленная t-статистика табличную. Для этого в качестве доверительного интервала выберем 95%
stats.t.ppf(0.975, 420)

midlle_e = np.mean(economy_array)
print(midlle_e)

midlle_c = np.mean(comfort_array_2)
print(midlle_c)

# расчитываем доверительный интервал
stats.t.interval(alpha=0.95, df=211, loc=np.mean(economy_array), scale=stats.sem(economy_array))

stats.t.interval(alpha=0.95, df=211, loc=np.mean(comfort_array_2), scale=stats.sem(comfort_array))

# Проанализую какая информация есть в отмененных поездках
# Создаю дату сожержащую только отмененные поездки
df_cancel = df[df['driver_response'] == 0]

Econ_cancel = df_cancel[df_cancel['offer_class_group'] == 'Economy']
Comf_cancel = df_cancel[df_cancel['offer_class_group'] == 'Comfort']
Deliv_cancel = df_cancel[df_cancel['offer_class_group'] == 'Delivery']
Prem_cancel = df_cancel[df_cancel['offer_class_group'] == 'Premium']

# Создадим списки разделив поездки по классам. Для того, чтобы потом построить график дистанции поездок по каждому классу.

Economy_cancel = list(Econ_cancel['distance_km'])
Comfort_cancel	= list(Comf_cancel['distance_km'])
Delivery_cancel = list(Deliv_cancel['distance_km'])
Premium_cancel	= list(Prem_cancel['distance_km'])

# Строим график распределения расстояний отмененных поездок по классам такси
plt.hist([Economy_cancel, Comfort_cancel, Delivery_cancel, Premium_cancel], bins = int(2000/200), color = colors, label=names)
plt.legend()
plt.xlabel("Расстояние поездок, км.")
plt.ylabel("Количество поездок")

# Создадим списки разделив поездки по классам. Для того, чтобы потом построить график по дням недели по каждому классу.
Economy_cancel_2 = list(Econ_cancel['weekday_key'])
Comfort_cancel_2	= list(Comf_cancel['weekday_key'])
Delivery_cancel_2 = list(Deliv_cancel['weekday_key'])
Premium_cancel_2	= list(Prem_cancel['weekday_key'])

# Строим график отмененных заказов по дня недели в зависимости от класса такси
plt.hist([Economy_cancel_2, Comfort_cancel_2, Delivery_cancel_2, Premium_cancel_2], bins = int(2000/200), color = colors, label=names)
plt.legend()
plt.xlabel("День недели")
plt.ylabel("Количество отмен")

# Построю график с разбивкой по дням и классам состоявшихся поездок

Economy_day = list(Econ['weekday_key'])
Comfort_day	= list(Comf['weekday_key'])
Delivery_day = list(Deliv['weekday_key'])
Premium_day	= list(Prem['weekday_key'])

plt.hist([Economy_day, Comfort_day, Delivery_day, Premium_day], bins = int(2000/200), color = colors, label=names)
plt.legend()
plt.xlabel("День недели")
plt.ylabel("Количество поездок")

# Хочу посмотреть есть ли зависимость отмены и поездок от времени суток

# Построю графики совершенных поездок и отмененных только в эконом-классе только в понедельник с разбивкой по часам.
df_economy = df[df['offer_class_group'] == 'Economy']
df_economy_1 = df_economy[df_economy['driver_response'] == 1]
df_economy_1_mon = df_economy_1[df_economy_1['weekday_key'] == 1]
df_economy_0 = df_economy[df_economy['driver_response'] == 0]
df_economy_0_mon = df_economy_0[df_economy_0['weekday_key'] == 1]
# df_economy_0_mon - отмененные заказы в экономе по понедельникам
# df-economy_1_mon - совершенные поездки в экономе по понедельникам
plt.hist(df_economy_0_mon['hour_key'], color = 'green', edgecolor = 'white')
plt.hist(df_economy_1_mon['hour_key'], color = 'orange', edgecolor = 'white')
plt.xlabel("Часы")
plt.ylabel("Количество заказов")

# Для сравнения построю аналогичные графики для комфорт-класса
df_comfort = df[df['offer_class_group'] == 'Comfort']
df_comfort_1 = df_comfort[df_comfort['driver_response'] == 1]
df_comfort_0 = df_comfort[df_comfort['driver_response'] == 0]
df_comfort_1_mon = df_comfort_1[df_comfort_1['weekday_key'] == 1]
df_comfort_0_mon = df_comfort_0[df_comfort_0['weekday_key'] == 1]

plt.hist(df_comfort_0_mon['hour_key'], color = 'green', edgecolor = 'white')
plt.hist(df_comfort_1_mon['hour_key'], color = 'orange', edgecolor = 'white')
plt.xlabel("Часы")
plt.ylabel("Количество заказов")

# Для сравнения построю аналогичные графики для премиум-класса

df_premium = df[df['offer_class_group'] == 'Premium']
df_premium_1 = df_premium[df_premium['driver_response'] == 1]
df_premium_0 = df_premium[df_premium['driver_response'] == 0]
df_premium_1_mon = df_premium_1[df_premium_1['weekday_key'] == 1]
df_premium_0_mon = df_premium_0[df_premium_0['weekday_key'] == 1]

plt.hist(df_premium_1_mon['hour_key'], color = 'orange', edgecolor = 'white')
plt.hist(df_premium_0_mon['hour_key'], color = 'green', edgecolor = 'white')
plt.xlabel("Часы")
plt.ylabel("Количество заказов")

