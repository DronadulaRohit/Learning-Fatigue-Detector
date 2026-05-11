import pandas as pd
import numpy as np
import random

def generate_study_sessions(n_samples=1000, output_file='study_sessions.csv'):
    """
    Generates synthetic study session data with embedded fatigue patterns.
    """
    np.random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        # Simulate different student profiles/situations
        profile = np.random.choice(['optimal', 'fatigued', 'distracted'], p=[0.4, 0.3, 0.3])
        
        if profile == 'optimal':
            # Optimal: Good duration, regular breaks, high focus, usually day time
            duration = np.random.normal(60, 15) # ~1 hour
            breaks = max(0, int(np.random.normal(duration / 25, 1))) # ~1 break per 25 mins
            time_of_day = np.random.randint(8, 20) # 8 AM to 8 PM
            time_since_last_break = np.random.uniform(0, 40)
            base_focus = np.random.normal(85, 5)
            
        elif profile == 'fatigued':
            # Fatigued: Long duration, few breaks, low focus, often late
            duration = np.random.normal(120, 30) # ~2 hours
            breaks = max(0, int(np.random.normal(1, 1))) # Very few breaks
            time_of_day = np.random.choice(list(range(20, 24)) + list(range(0, 4))) # Late night
            time_since_last_break = np.random.uniform(60, 120) # Long time since break
            base_focus = np.random.normal(40, 10)
            
        else: # distracted
            # Distracted: Short bursts, too many breaks, variable focus
            duration = np.random.normal(30, 10)
            breaks = max(1, int(np.random.normal(3, 1))) # Many breaks
            time_of_day = np.random.randint(10, 22)
            time_since_last_break = np.random.uniform(0, 10)
            base_focus = np.random.normal(60, 10)
        
        # Clip values to realistic ranges
        duration = max(10, duration)
        focus_score = np.clip(base_focus, 0, 100)
        
        data.append({
            'session_duration_min': round(duration, 1),
            'break_frequency': breaks,
            'focus_score': round(focus_score, 1),
            'time_of_day': time_of_day,
            'time_since_last_break': round(time_since_last_break, 1),
            'profile_ground_truth': profile # For validation/debugging, can be dropped later
        })
        
    df = pd.DataFrame(data)
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Data generated successfully and saved to {output_file}")
        
    return df

if __name__ == "__main__":
    generate_study_sessions()
