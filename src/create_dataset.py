import pandas as pd
import numpy as np

# Počet riadkov
n = 500

# Numerické stĺpce
age = np.random.randint(1, 80, size=n)                 # vek 1–79
fare = np.round(np.random.uniform(5, 150, size=n), 2)  # cena lístka

# Kategóriálne stĺpce
sex = np.random.choice(['male', 'female'], size=n)
pclass = np.random.choice(['first', 'second', 'third'], size=n)

# Cieľová premenná (survived) s realistickými pravidlami
survived = []
for i in range(n):
    prob = 0.2  # základná pravdepodobnosť prežitia
    if sex[i] == 'female':
        prob += 0.3
    if pclass[i] == 'first':
        prob += 0.2
    elif pclass[i] == 'second':
        prob += 0.1
    # vek: malé deti majú vyššiu šancu prežitia
    if age[i] <= 10:
        prob += 0.1
    # fare: vyššie lístky môžu naznačovať vyššiu šancu
    if fare[i] > 100:
        prob += 0.05

    survived.append(np.random.choice([0, 1], p=[1-prob, prob]))

# Vytvorenie DataFrame
df = pd.DataFrame({
    'age': age,
    'sex': sex,
    'fare': fare,
    'class': pclass,
    'survived': survived
})

# Uloženie do CSV
df.to_csv('../data/dataset.csv', index=False)

print("Realistický dataset.csv bol vytvorený s", n, "riadkami.")
