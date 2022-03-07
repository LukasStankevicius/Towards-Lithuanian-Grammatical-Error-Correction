from errant_wrapper import tuples_to_scores
import pandas as pd

# Input 1-3: True positives, false positives, false negatives
# Input 4: Value of beta in F-score.
# Output 1-3: Precision, Recall and F-score rounded to 4dp.
def computeFScore(tp, fp, fn, beta=0.5):
    p = float(tp)/(tp+fp) if fp else 1.0
    r = float(tp)/(tp+fn) if fn else 1.0
    f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def test_tuples_to_scores():
    #  [("bad sentence", "hypothesis sentence", "reference sentence"), ...]
    samples = [['VILNIUS, ruodžio20 – Sputnik. Lietuvos bankas vyk/o terimą dėl praėjusiais metias žlugusios Vokietijos kompanijos„Wirecard“ pinigų plovim o skandalel galimai dalyvavutios Lituvoje registruo tos fin tech įmoės „Finolita Unio“, praneša bainko sp audos tarnybą',
 'VILNIUS, gruodžio 20 – Sputnik. Lietuvos bankas vykdo tyrimą dėl praėjusiais metais žlugusios Vokietijos kompanijos „Wirecard“ pinigų plovimo skandale galimai dalyvavusios Lietuvoje registruotos fintech įmonės „Finolita Unio“, praneša banko spaudos tarnyba.',
 'VILNIUS, gruodžio 20 – Sputnik. Lietuvos bankas vykdo tyrimą dėl praėjusiais metais žlugusios Vokietijos kompanijos „Wirecard“ pinigų plovimo skandale galimai dalyvavusios Lietuvoje registruotos fintech įmonės „Finolita Unio“, praneša banko spaudos tarnyba'],
               [
                   '„Manchester City“ paiksiantis futbo lininkas Ya ya Toure tegailėjo aštrių ir griežtų žodžiųvyriausiajam koma ndos treneri ui PepuiGuardiolai.',
                   '„Manchester City“ baigsiantis futbolininkas Yaya Toure tegailėjo aštrių ir griežtų žodžių vyriausiajam komandos treneriui Pepui Guardiolai.',
                   '„Manchester City“ paliksiantis futbolininkas Yaya Toure negailėjo aštrių ir griežtų žodžių vyriausiajam komandos treneriui Pepui Guardiolai.'],
               [
                   '– Vedant ekskursijas galime Išvardyti datas, supažindinti su faktais ir tiek. Tačiau žmones traųkia kitokie dalykai, kartais – paslaptingi, net abisūes. Kodėl aip yra?',
                   '– Vedant ekskursijas galime išvardyti datas, supažindinti su faktais ir tiek. Tačiau žmones traukia kitokie dalykai, kartais – paslaptingi, net abi sūnės. Kodėl taip yra?',
                   '– Vedant ekskursijas galima išvardyti datas, supažindinti su faktais ir tiek. Tačiau žmones traukia kitokie dalykai, kartais – paslaptingi, net baisūs. Kodėl taip yra?'],
               [
                   'Šiem et įpri valomąją karotarnybą pašaukti 3 tūkstančiai jaunuolių. Kitąmet šaukimą planuojama pratėti sausio 8 deną.',
                   'Šiemet į privalomąją karo tarnybą pašaukti 3 tūkstančiai jaunuolių. Kitąmet šaukimą planuojama pradėti sausio 8 dieną.',
                   'Šiemet į privalomąją karo tarnybą pašaukti 3 tūkstančiai jaunuolių. Kitąmet šaukimą planuojama pradėti sausio 7 dieną.'],
               ['Migrena gali prišaukti insultą',
                'Migrena gali prišaukti insultą',
                'Migrena gali prišaukti insultą'],
               ['Gruodžio 6, 2 ir 17 dienomis Seimo posėdžiai nebuvo rengia mi(',
                'Gruodžio 6, 2 ir 17 dienomis Seimo posėdžiai nebuvo rengiami.',
                'Gruodžio 6, 7 ir 17 dienomis Seimo posėdžiai nebuvo rengiami.'],
               [
                   'Jis pasiryžęs vaistų vykti pirkti į kaimynines šalis, ma kiti anksčiau vartoit medikamenta i mamai sukelsdavo alergiją: „Eksperimentuoti su sveikata nesinori.“',
                   'Jis pasiryžęs vaistų vykti pirkti į kaimynines šalis, mat kiti anksčiau vartoti medikamentai mamai sukeldavo alergiją: „Eksperimentuoti su sveikata nesinori.“',
                   'Jis pasiryžęs vaistų vykti pirkti į kaimynines šalis, mat kiti anksčiau vartoti medikamentai mamai sukeldavo alergiją: „Eksperimentuoti su sveikata nesinori.“'],
               ["Jis, ji., mes: gal .",
                "Jis, ji, mes: gal.",
                "Jis, ji, mes: gal..."]
               ]
    answ, scores = tuples_to_scores(samples)
    df = pd.DataFrame(scores, columns=['tp', 'fp', 'fn']).sum()
    fscore = computeFScore(df['tp'], df['fp'], df['fn'])
    assert fscore == (0.9091, 0.8108, 0.8876)
