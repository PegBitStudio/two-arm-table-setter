# Testing on an Intel Core Ultra laptop — volunteer guide

Thank you for helping! This runs our robot project on your laptop and measures how fast Intel's
OpenVINO runs it on your processor, graphics and NPU. Nothing is sent anywhere automatically:
at the end you get one zip file to send back.

**You need:** a Windows 11 laptop with an **Intel Core Ultra** processor (Series 2 or 3 is ideal),
about **8 GB of free disk space**, a good internet connection (about 3.5 GB of downloads),
and **about an hour**. Keep the laptop **plugged in**.

Check your processor: **Settings → System → About → Processor** should say "Intel Core Ultra".

---

## 1. Install Python and Git (skip if you have them)

Open **PowerShell** (press Start, type `powershell`, press Enter) and paste:

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
```

Then **close PowerShell and open it again** so it finds them.

## 2. Download the project and its libraries

Paste these one at a time (the last one takes 5–15 minutes):

```powershell
cd $HOME
git clone https://github.com/PegBitStudio/two-arm-table-setter.git
cd two-arm-table-setter
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

## 3. Stop the laptop from sleeping during the test

**Settings → System → Power & battery → Screen and sleep** → set "When plugged in, put my device
to sleep after" to **Never** for now. (You can change it back afterwards.)

## 4. Run the test — one command

```powershell
.venv\Scripts\python bench\volunteer.py
```

It will:

1. List the Intel chips OpenVINO can use — you should see **CPU**, **GPU** and, on Core Ultra, **NPU**.
2. Download the AI model (3.1 GB) and our trained robot policy (45 MB).
3. Measure speed on each chip (the longest part).
4. Let the two robot arms set 10 random tables.
5. Create **`volunteer_results.zip`** in the `two-arm-table-setter` folder.

Small windows may flash up briefly — that's the simulator drawing camera pictures. Let it finish.

## 5. Send back the results

Send **`volunteer_results.zip`** (it's small) by WhatsApp, email or Google Drive.
It contains only speed numbers, your processor/graphics names, Windows version and the test log —
no personal files.

---

## If something goes wrong

| Problem | What to do |
|---|---|
| `winget` not found | Install Python 3.12 from python.org (tick "Add to PATH") and Git from git-scm.com |
| `py -3.12` not found | Close and reopen PowerShell; or use the full path to `python.exe` from the Python install |
| No **NPU** in step 1 | Fine — it still runs on CPU and GPU. To add the NPU, update the Intel NPU driver (Windows Update → Advanced options → Optional updates, or intel.com "Intel NPU driver") and run step 4 again |
| It seems stuck | The AI-model part can sit quiet for a few minutes while it loads. Give it 10 minutes before stopping |
| Anything else | Send a screenshot of the error plus the `volunteer_results` folder |

Want a quick 10-minute check first? Run `.venv\Scripts\python bench\volunteer.py --quick`.

**To clean up afterwards:** delete the `two-arm-table-setter` folder and the `.models` folder in
your user folder (`C:\Users\<you>\.models`).
