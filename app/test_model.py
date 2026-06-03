"""Symmetry and sanity test for the overhauled model."""
import pickle
from ufc_predictor import predict_hypothetical_fight

model = pickle.load(open('ufc_logistic_model.pkl', 'rb'))
arts = pickle.load(open('ufc_other_artifacts.pkl', 'rb'))
df = arts['data_for_lookups']
cols = arts['numerical_features'] + arts['categorical_features']

print("=" * 60)
print("SYMMETRY TEST")
print("=" * 60)

r1 = predict_hypothetical_fight('Conor McGregor', 'Khabib Nurmagomedov', model, df, cols)
print()
r2 = predict_hypothetical_fight('Khabib Nurmagomedov', 'Conor McGregor', model, df, cols)
print()

mcg_prob_1 = r1['red_win_prob']
mcg_prob_2 = r2['blue_win_prob']
khabib_prob_1 = r1['blue_win_prob']
khabib_prob_2 = r2['red_win_prob']
print("--- SYMMETRY RESULT ---")
print(f"McGregor(Red) vs Khabib(Blue): McGregor = {mcg_prob_1*100:.1f}%, Khabib = {khabib_prob_1*100:.1f}%")
print(f"Khabib(Red) vs McGregor(Blue): McGregor = {mcg_prob_2*100:.1f}%, Khabib = {khabib_prob_2*100:.1f}%")
print(f"McGregor prob difference: {abs(mcg_prob_1 - mcg_prob_2)*100:.2f} pp (should be 0)")
print(f"Khabib prob difference: {abs(khabib_prob_1 - khabib_prob_2)*100:.2f} pp (should be 0)")

print()
print("=" * 60)
print("SANITY CHECKS")
print("=" * 60)

fights = [
    ('Israel Adesanya', 'Sean Strickland'),
    ('Alex Pereira', 'Ciryl Gane'),
    ('Islam Makhachev', 'Charles Oliveira'),
    ('Jon Jones', 'Stipe Miocic'),
    ('Sean O\'Malley', 'Marlon Vera'),
]

for f1, f2 in fights:
    print()
    predict_hypothetical_fight(f1, f2, model, df, cols)
