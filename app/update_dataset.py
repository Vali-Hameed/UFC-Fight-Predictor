import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from datetime import datetime
import os
import re

CSV_FILE = 'ufc-master.csv'

def convert_height(height_str):
    if not height_str or "--" in height_str:
        return None
    try:
        parts = height_str.replace('"', '').split("'")
        feet = int(parts[0].strip())
        inches = int(parts[1].strip())
        return round((feet * 12 + inches) * 2.54, 2)
    except:
        return None

def convert_reach(reach_str):
    if not reach_str or "--" in reach_str:
        return None
    try:
        inches = float(reach_str.replace('"', '').strip())
        return round(inches * 2.54, 2)
    except:
        return None

def calculate_age(dob_str, fight_date):
    if not dob_str or "--" in dob_str:
        return None
    try:
        dob = datetime.strptime(dob_str, "%b %d, %Y")
        age_years = fight_date.year - dob.year - ((fight_date.month, fight_date.day) < (dob.month, dob.day))
        return age_years
    except:
        return None

def scrape_fighter_stats(page, fighter_url, fight_date):
    page.goto(fighter_url)
    page.wait_for_selector('.b-list__info-box-left', timeout=30000)
    soup = BeautifulSoup(page.content(), 'html.parser')
    
    stats = {
        "Wins": 0, "Losses": 0, "Draws": 0,
        "WinsByKO": 0, "WinsBySubmission": 0,
        "CurrentWinStreak": 0, "CurrentLoseStreak": 0, "LongestWinStreak": 0,
        "TotalRoundsFought": 0, "TotalTitleBouts": 0,
        "AvgSigStrLanded": 0.0, "AvgSigStrPct": 0.0, 
        "AvgTDLanded": 0.0, "AvgTDPct": 0.0, "AvgSubAtt": 0.0,
        "Stance": "Orthodox", "HeightCms": None, "ReachCms": None, "Age": None
    }
    
    # Career Stats
    info_items = soup.select('.b-list__box-list-item')
    dob_str = None
    for item in info_items:
        text = item.text.strip().replace('  ', '').replace('\n', '')
        if "Height:" in text:
            stats['HeightCms'] = convert_height(text.replace("Height:", "").strip())
        elif "Reach:" in text:
            stats['ReachCms'] = convert_reach(text.replace("Reach:", "").strip())
        elif "STANCE:" in text:
            stats['Stance'] = text.replace("STANCE:", "").strip()
        elif "DOB:" in text:
            dob_str = text.replace("DOB:", "").strip()
        elif "SLpM:" in text:
            val = text.replace("SLpM:", "").strip()
            stats['AvgSigStrLanded'] = float(val) if val != '--' else 0.0
        elif "Str. Acc.:" in text:
            val = text.replace("Str. Acc.:", "").replace("%", "").strip()
            stats['AvgSigStrPct'] = float(val)/100.0 if val != '--' else 0.0
        elif "TD Avg.:" in text:
            val = text.replace("TD Avg.:", "").strip()
            stats['AvgTDLanded'] = float(val) if val != '--' else 0.0
        elif "TD Acc.:" in text:
            val = text.replace("TD Acc.:", "").replace("%", "").strip()
            stats['AvgTDPct'] = float(val)/100.0 if val != '--' else 0.0
        elif "Sub. Avg.:" in text:
            val = text.replace("Sub. Avg.:", "").strip()
            stats['AvgSubAtt'] = float(val) if val != '--' else 0.0

    stats['Age'] = calculate_age(dob_str, fight_date)

    # History (only count fights BEFORE this fight_date)
    history_rows = soup.select('.b-fight-details__table-row.b-fight-details__table-row__hover.js-fight-details-click')
    
    current_win_streak = 0
    current_lose_streak = 0
    longest_win_streak = 0
    temp_win_streak = 0
    streak_broken = False
    
    # Rows are ordered from newest to oldest
    for row in history_rows:
        cols = row.select('td')
        if not cols or len(cols) < 10:
            continue
            
        event_col = cols[6]
        event_date_str = event_col.select('p')[-1].text.strip() if len(event_col.select('p')) > 1 else ""
        try:
            event_date = datetime.strptime(event_date_str, "%b. %d, %Y")
        except:
            continue
            
        # Only count stats from BEFORE the current fight
        if event_date >= fight_date:
            continue
            
        result_flag = cols[0].select_one('.b-flag__text')
        result = result_flag.text.strip().lower() if result_flag else ""
        
        method_p = cols[7].select_one('p')
        method = method_p.text.strip().lower() if method_p else ""
        
        rnd_p = cols[8].select_one('p')
        if rnd_p and rnd_p.text.strip().isdigit():
            stats["TotalRoundsFought"] += int(rnd_p.text.strip())
            
        # Heuristic for title bouts (5 rounds scheduled - usually indicated by a belt image, but we'll check if it's 5 rounds or belt icon)
        images = cols[6].select('img')
        is_title = any('belt' in img.get('src', '').lower() for img in images)
        if is_title:
            stats["TotalTitleBouts"] += 1
            
        if result == 'win':
            stats['Wins'] += 1
            temp_win_streak += 1
            if temp_win_streak > longest_win_streak:
                longest_win_streak = temp_win_streak
            
            if not streak_broken:
                current_win_streak += 1
            else:
                streak_broken = True
                
            if "ko" in method or "tko" in method:
                stats['WinsByKO'] += 1
            elif "sub" in method:
                stats['WinsBySubmission'] += 1
                
        elif result == 'loss':
            stats['Losses'] += 1
            temp_win_streak = 0
            if not streak_broken:
                current_lose_streak += 1
                streak_broken = True
        elif result == 'draw' or result == 'nc':
            stats['Draws'] += 1
            temp_win_streak = 0
            streak_broken = True

    stats['CurrentWinStreak'] = current_win_streak
    stats['CurrentLoseStreak'] = current_lose_streak
    stats['LongestWinStreak'] = longest_win_streak
    
    return stats

def run_update():
    print(f"Loading {CSV_FILE} to find the latest fight date...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"{CSV_FILE} not found. Ensure you run this from the app directory.")
        return
        
    df['Date'] = pd.to_datetime(df['Date'])
    latest_date_in_csv = df['Date'].max()
    print(f"Latest fight date in CSV: {latest_date_in_csv.strftime('%Y-%m-%d')}")
    
    new_rows = []
    columns = df.columns.tolist()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Scrape completed events
        page.goto("http://ufcstats.com/statistics/events/completed?page=all")
        page.wait_for_selector('.b-statistics__table-events', timeout=30000)
        soup = BeautifulSoup(page.content(), 'html.parser')
        
        events = []
        rows = soup.select('.b-statistics__table-events tr.b-statistics__table-row')
        for row in rows[1:]:
            columns_td = row.select('td')
            if not columns_td: continue
            
            link_tag = columns_td[0].select_one('a')
            if not link_tag: continue
            
            date_tag = columns_td[0].select_one('.b-statistics__date')
            date_str = date_tag.text.strip() if date_tag else None
            if not date_str: continue
            
            try:
                event_date = datetime.strptime(date_str, "%B %d, %Y")
                if event_date > latest_date_in_csv:
                    events.append({
                        "name": link_tag.text.strip(),
                        "url": link_tag['href'],
                        "date": event_date,
                        "location": columns_td[1].text.strip().replace('\n', ' ') if len(columns_td) > 1 else ""
                    })
            except:
                pass
                
        events.sort(key=lambda x: x['date']) # Oldest to newest new events
        
        if not events:
            print("No new events found to append. The CSV is up to date!")
            browser.close()
            return
            
        print(f"Found {len(events)} new events to process.")
        
        for event in events:
            print(f"Processing Event: {event['name']} ({event['date'].strftime('%Y-%m-%d')})")
            page.goto(event['url'])
            page.wait_for_selector('.b-fight-details__table', timeout=30000)
            event_soup = BeautifulSoup(page.content(), 'html.parser')
            
            fights = event_soup.select('.b-fight-details__table-row')
            for fight_row in fights[1:]:
                cols = fight_row.select('td')
                if not cols or len(cols) < 10: continue
                
                fighters = cols[1].select('a')
                if len(fighters) < 2: continue
                
                # In ufcstats, fighter 1 (Red corner) is usually the winner if win badges match, else we just assign them
                win_badges = cols[0].select('.b-flag__text')
                f1_win = "win" in win_badges[0].text.strip().lower() if len(win_badges) > 0 else False
                f2_win = "win" in win_badges[1].text.strip().lower() if len(win_badges) > 1 else False
                
                if not f1_win and not f2_win:
                    continue # Ignore draws/NC for training usually, or just assign arbitrarily
                
                red_fighter_name = fighters[0].text.strip()
                blue_fighter_name = fighters[1].text.strip()
                winner = "Red" if f1_win else "Blue"
                
                weight_class = cols[6].text.strip()
                is_title_bout = "title" in weight_class.lower()
                
                f1_url = fighters[0]['href']
                f2_url = fighters[1]['href']
                
                print(f"  Scraping fight: {red_fighter_name} vs {blue_fighter_name}")
                red_stats = scrape_fighter_stats(page, f1_url, event['date'])
                blue_stats = scrape_fighter_stats(page, f2_url, event['date'])
                
                # Create a new row matching the CSV format
                new_row = {col: None for col in columns}
                new_row['RedFighter'] = red_fighter_name
                new_row['BlueFighter'] = blue_fighter_name
                new_row['Date'] = event['date'].strftime('%Y-%m-%d')
                new_row['Location'] = event['location']
                new_row['Winner'] = winner
                new_row['TitleBout'] = str(is_title_bout)
                new_row['WeightClass'] = weight_class
                
                # Map Red Stats
                new_row['RedAge'] = red_stats['Age']
                new_row['RedHeightCms'] = red_stats['HeightCms']
                new_row['RedReachCms'] = red_stats['ReachCms']
                new_row['RedStance'] = red_stats['Stance']
                new_row['RedCurrentWinStreak'] = red_stats['CurrentWinStreak']
                new_row['RedCurrentLoseStreak'] = red_stats['CurrentLoseStreak']
                new_row['RedLosses'] = red_stats['Losses']
                new_row['RedTotalRoundsFought'] = red_stats['TotalRoundsFought']
                new_row['RedTotalTitleBouts'] = red_stats['TotalTitleBouts']
                new_row['RedWinsByKO'] = red_stats['WinsByKO']
                new_row['RedWinsBySubmission'] = red_stats['WinsBySubmission']
                new_row['RedAvgTDLanded'] = red_stats['AvgTDLanded']
                new_row['RedAvgSubAtt'] = red_stats['AvgSubAtt']
                new_row['RedAvgSigStrLanded'] = red_stats['AvgSigStrLanded']
                new_row['RedAvgSigStrPct'] = red_stats['AvgSigStrPct']
                
                # Map Blue Stats
                new_row['BlueAge'] = blue_stats['Age']
                new_row['BlueHeightCms'] = blue_stats['HeightCms']
                new_row['BlueReachCms'] = blue_stats['ReachCms']
                new_row['BlueStance'] = blue_stats['Stance']
                new_row['BlueCurrentWinStreak'] = blue_stats['CurrentWinStreak']
                new_row['BlueCurrentLoseStreak'] = blue_stats['CurrentLoseStreak']
                new_row['BlueLosses'] = blue_stats['Losses']
                new_row['BlueTotalRoundsFought'] = blue_stats['TotalRoundsFought']
                new_row['BlueTotalTitleBouts'] = blue_stats['TotalTitleBouts']
                new_row['BlueWinsByKO'] = blue_stats['WinsByKO']
                new_row['BlueWinsBySubmission'] = blue_stats['WinsBySubmission']
                new_row['BlueAvgTDLanded'] = blue_stats['AvgTDLanded']
                new_row['BlueAvgSubAtt'] = blue_stats['AvgSubAtt']
                new_row['BlueAvgSigStrLanded'] = blue_stats['AvgSigStrLanded']
                new_row['BlueAvgSigStrPct'] = blue_stats['AvgSigStrPct']
                
                # Calculate Difs
                new_row['HeightDif'] = (red_stats['HeightCms'] or 0) - (blue_stats['HeightCms'] or 0)
                new_row['ReachDif'] = (red_stats['ReachCms'] or 0) - (blue_stats['ReachCms'] or 0)
                new_row['WinStreakDif'] = red_stats['CurrentWinStreak'] - blue_stats['CurrentWinStreak']
                new_row['LossDif'] = red_stats['Losses'] - blue_stats['Losses']
                new_row['TotalRoundDif'] = red_stats['TotalRoundsFought'] - blue_stats['TotalRoundsFought']
                new_row['TotalTitleBoutDif'] = red_stats['TotalTitleBouts'] - blue_stats['TotalTitleBouts']
                new_row['KODif'] = red_stats['WinsByKO'] - blue_stats['WinsByKO']
                new_row['SubDif'] = red_stats['WinsBySubmission'] - blue_stats['WinsBySubmission']
                new_row['AvgTDDif'] = red_stats['AvgTDLanded'] - blue_stats['AvgTDLanded']
                new_row['AvgSubAttDif'] = red_stats['AvgSubAtt'] - blue_stats['AvgSubAtt']
                
                new_rows.append(new_row)
                
        browser.close()
        
    if new_rows:
        print(f"Appending {len(new_rows)} new fights to {CSV_FILE}...")
        new_df = pd.DataFrame(new_rows)
        new_df = new_df[columns] # Ensure column order matches exactly
        new_df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        print("Done!")

if __name__ == "__main__":
    run_update()
