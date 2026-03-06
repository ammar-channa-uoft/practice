
import pandas as pd
import numpy as np

MEGA_PWRSP_PKL = "/home/achanna/projects/rrg-bojana/achanna/kylie_combined_picklefiles/megapwrspdf_unresampled.pkl"

micetoexclude = [['2024-08-22', 'C'],
                 ['2024-08-22', 'S'],
                 ['2024-08-01', 'C'],
                 ['2024-07-26', 'C']]

micetoexclude = pd.DataFrame(micetoexclude, columns = ['date','stim'])

powspecfrqs = [0.5, 4, 8, 12, 27, 80, 140]
powspecbands = ['δ', 'θ', 'α', 'β', 'low-γ', 'high-γ']

megapwrspdf = pd.read_pickle(MEGA_PWRSP_PKL)


indexbad = megapwrspdf[(megapwrspdf['date'] == '2024-07-10') & (megapwrspdf['subtime'] == '163831')].index
megapwrspdf.drop(indexbad, inplace=True)
megapwrspdf = megapwrspdf.reset_index(drop=True)

indexbad = megapwrspdf[(megapwrspdf['date'] == '2024-08-09') & (megapwrspdf['subtime'] == '152116')].index
megapwrspdf.drop(indexbad, inplace=True)
megapwrspdf = megapwrspdf.reset_index(drop=True)


indexAge = megapwrspdf[megapwrspdf['tp']*2 >= 20].index
megapwrspdf.drop(indexAge, inplace=True)
megapwrspdf = megapwrspdf.reset_index(drop=True)


indexelectrode = megapwrspdf[megapwrspdf['electrode'] > 14].index
megapwrspdf.drop(indexelectrode, inplace=True)
megapwrspdf = megapwrspdf.reset_index(drop=True)

indexelectrode = megapwrspdf[megapwrspdf['electrode'] < 1].index
megapwrspdf.drop(indexelectrode, inplace=True)
megapwrspdf = megapwrspdf.reset_index(drop=True)


merged = megapwrspdf.merge(micetoexclude, on=['date', 'stim'], how='left', indicator=True)
megapwrspdf = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
megapwrspdf = megapwrspdf.reset_index(drop=True)

megapwrspdf = megapwrspdf[megapwrspdf["resampled"] == False].reset_index(drop=True)


rows = []

for date in megapwrspdf["date"].unique():
    # loop over electrodes present in unresampled data
    for electrode in sorted(megapwrspdf["electrode"].unique()):
        for stim in sorted(megapwrspdf["stim"].unique()):
            tolook = megapwrspdf[
                (megapwrspdf["date"] == date) &
                (megapwrspdf["electrode"] == electrode) &
                (megapwrspdf["stim"] == stim)]

            if len(tolook) == 0:
                continue

            # We export mean power separately for each hemisphere
            for hemi in ["ipsi", "contra"]:
                tolook_hemi = tolook[tolook["hemi"] == hemi].reset_index(drop=True)
                if len(tolook_hemi) == 0:
                    continue

                # compute bandpower for each band
                for b in range(len(powspecfrqs) - 1):
                    avpwr = []
                    for i in range(len(tolook_hemi)):
                        freqtemp = tolook_hemi.loc[i]["freq"]
                        pwrsptemp = np.mean(tolook_hemi.loc[i]["pwrsp"], axis=0)

                        # drop 60 Hz band
                        freqtodrop = np.where((freqtemp >= 59) & (freqtemp <= 61))[0]
                        freqtemp = np.delete(freqtemp, freqtodrop)
                        pwrsptemp = np.delete(pwrsptemp, freqtodrop)

                        # drop 120 Hz band
                        freqtodrop = np.where((freqtemp >= 119) & (freqtemp <= 121))[0]
                        freqtemp = np.delete(freqtemp, freqtodrop)
                        pwrsptemp = np.delete(pwrsptemp, freqtodrop)

                        band_mask = np.where((freqtemp >= powspecfrqs[b]) & (freqtemp < powspecfrqs[b+1]))[0]
                        if len(band_mask) == 0:
                            continue

                        avpwr.append(pwrsptemp[band_mask])

                    if len(avpwr) == 0:
                        continue

                    mean_power = float(np.mean(np.array(avpwr)))

                    # grab metadata from this session chunk
                    # (these should be constant within tolook)
                    subtime = tolook["subtime"].unique()[0] if "subtime" in tolook.columns else None
                    dps = tolook["dps"].unique()[0] if "dps" in tolook.columns else None
                    sex = tolook["sex"].unique()[0] if "sex" in tolook.columns else None

                    rows.append({
                        "channel": electrode,
                        "date": date,
                        "subtime": subtime,
                        "mouse_session": f"{date}_{subtime}" if subtime is not None else str(date),
                        "hemisphere": hemi,
                        "days_post_stroke": dps,
                        "stim": stim,
                        "freq_band": powspecbands[b],
                        "freq_low_hz": powspecfrqs[b],
                        "freq_high_hz": powspecfrqs[b+1],
                        "mean_power": mean_power,
                        "sex": sex})

outdf = pd.DataFrame(rows)

out_csv = "bandpower_table_unresampled.csv"
outdf.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("Rows:", len(outdf))
print(outdf.head(10))
