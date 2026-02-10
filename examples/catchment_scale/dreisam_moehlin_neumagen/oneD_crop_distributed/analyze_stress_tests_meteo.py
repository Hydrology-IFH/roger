from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def read_meteo(path_to_dir: Path):
    """Reads the meteorological data

    Data is imported from .txt files and stored in dataframes. Format of NA/NaN
    values is -9999.

    Args
    ----------
    path_to_dir : Path
        path to directions which contains input data

    Returns
    ----------
    prec_10mins : pd.DataFrame
        precipitation (in mm/10 mins)

    df_ta : pd.DataFrame
        air temperature (in °C)

    df_pet : pd.DataFrame
        potential evapotranspiration (in mm/day)
    """
    if not os.path.isdir(path_to_dir):
        raise ValueError(path_to_dir, "does not exist")

    Ta_path = path_to_dir / "TA.txt"
    PREC_path = path_to_dir / "PREC.txt"
    PET_path = path_to_dir / "PET.txt"

    df_PREC = pd.read_csv(
        PREC_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        na_values=-9999,
    )
    df_PREC.index = pd.to_datetime(dict(year=df_PREC.YYYY, month=df_PREC.MM, day=df_PREC.DD, hour=df_PREC.hh, minute=df_PREC.mm))
    df_PREC = df_PREC.loc[:, ["PREC"]]
    df_PREC.index = df_PREC.index.rename("Index")

    if os.path.exists(PET_path):
        df_pet = pd.read_csv(
            PET_path,
            sep=r"\s+",
            skiprows=0,
            header=0,
            na_values=-9999,
        )
        df_pet.index = pd.to_datetime(dict(year=df_pet.YYYY, month=df_pet.MM, day=df_pet.DD, hour=df_pet.hh, minute=df_pet.mm))
        df_pet = df_pet.loc[:, ["PET"]]
        df_pet.index = df_pet.index.rename("Index")
    else:
        df_pet = None

    df_ta = pd.read_csv(
        Ta_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        na_values=-9999,
    )
    df_ta.index = pd.to_datetime(dict(year=df_ta.YYYY, month=df_ta.MM, day=df_ta.DD, hour=df_ta.hh, minute=df_ta.mm))
    df_ta = df_ta.loc[:, ["TA"]]
    df_ta.index = df_ta.index.rename("Index")

    return df_PREC, df_pet, df_ta


def aggregate_to_monthly(df_prec, df_ta, df_pet):
    """Aggregate data to monthly values"""
    # Precipitation: sum
    prec_monthly = df_prec.resample('ME').sum()
    
    # Temperature: mean
    ta_monthly = df_ta.resample('ME').mean()
    
    # PET: sum
    pet_monthly = df_pet.resample('ME').sum()
    
    return prec_monthly, ta_monthly, pet_monthly


def aggregate_to_seasonal(df_prec, df_ta, df_pet):
    """Aggregate data to seasonal values
    Spring: MAM (March-May)
    Summer: JJA (June-August)
    Autumn: SON (September-November)
    Winter: DJF (December-February)
    """
    # Precipitation: sum
    prec_seasonal = df_prec.resample('QE-NOV').sum()
    prec_seasonal = prec_seasonal[1:-1]  # Remove incomplete seasons at start and end)]
    
    # Temperature: mean
    ta_seasonal = df_ta.resample('QE-NOV').mean()
    ta_seasonal = ta_seasonal[1:-1]  # Remove incomplete seasons at start and end
    
    # PET: sum
    pet_seasonal = df_pet.resample('QE-NOV').sum()
    pet_seasonal = pet_seasonal[1:-1]  # Remove incomplete seasons at start and end
    
    return prec_seasonal, ta_seasonal, pet_seasonal


def aggregate_to_annual(df_prec, df_ta, df_pet):
    """Aggregate data to annual values"""
    # Precipitation: sum
    prec_annual = df_prec.resample('YE').sum()
    
    # Temperature: mean
    ta_annual = df_ta.resample('YE').mean()
    
    # PET: sum
    pet_annual = df_pet.resample('YE').sum()
    
    return prec_annual, ta_annual, pet_annual


def plot_comparisons(base_data, stress_data, scenario_name, station, duration, magnitude, figures_dir):
    """Create comparison plots for base and stress scenarios"""
    
    # Unpack data
    (prec_mon_base, ta_mon_base, pet_mon_base,
     prec_seas_base, ta_seas_base, pet_seas_base,
     prec_ann_base, ta_ann_base, pet_ann_base) = base_data
    
    (prec_mon_stress, ta_mon_stress, pet_mon_stress,
     prec_seas_stress, ta_seas_stress, pet_seas_stress,
     prec_ann_stress, ta_ann_stress, pet_ann_stress) = stress_data
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f'{scenario_name} - Station {station} - Duration {duration} - Magnitude {magnitude}', 
                 fontsize=16, fontweight='bold')
    
    # Monthly plots
    # Precipitation
    axes[0, 0].plot(prec_mon_base.index, prec_mon_base['PREC'], label='Baseline', linewidth=1.5, alpha=0.7)
    axes[0, 0].plot(prec_mon_stress.index, prec_mon_stress['PREC'], label='Stress', linewidth=1.5, alpha=0.7)
    axes[0, 0].set_title('Monthly Precipitation', fontweight='bold')
    axes[0, 0].set_ylabel('Precipitation (mm/month)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Temperature
    axes[0, 1].plot(ta_mon_base.index, ta_mon_base['TA'], label='Baseline', linewidth=1.5, alpha=0.7)
    axes[0, 1].plot(ta_mon_stress.index, ta_mon_stress['TA'], label='Stress', linewidth=1.5, alpha=0.7)
    axes[0, 1].set_title('Monthly Temperature', fontweight='bold')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PET
    axes[0, 2].plot(pet_mon_base.index, pet_mon_base['PET'], label='Baseline', linewidth=1.5, alpha=0.7)
    axes[0, 2].plot(pet_mon_stress.index, pet_mon_stress['PET'], label='Stress', linewidth=1.5, alpha=0.7)
    axes[0, 2].set_title('Monthly PET', fontweight='bold')
    axes[0, 2].set_ylabel('PET (mm/month)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Seasonal plots
    # Precipitation
    axes[1, 0].plot(prec_seas_base.index, prec_seas_base['PREC'], label='Baseline', linewidth=1.5, alpha=0.7)
    axes[1, 0].plot(prec_seas_stress.index, prec_seas_stress['PREC'], label='Stress', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_title('Seasonal Precipitation', fontweight='bold')
    axes[1, 0].set_ylabel('Precipitation (mm/season)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Temperature
    axes[1, 1].plot(ta_seas_base.index, ta_seas_base['TA'], label='Baseline', linewidth=1.5, alpha=0.7)
    axes[1, 1].plot(ta_seas_stress.index, ta_seas_stress['TA'], label='Stress', linewidth=1.5, alpha=0.7)
    axes[1, 1].set_title('Seasonal Temperature', fontweight='bold')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # PET
    axes[1, 2].plot(pet_seas_base.index, pet_seas_base['PET'], label='Baseline', linewidth=1.5, alpha=0.7)
    axes[1, 2].plot(pet_seas_stress.index, pet_seas_stress['PET'], label='Stress', linewidth=1.5, alpha=0.7)
    axes[1, 2].set_title('Seasonal PET', fontweight='bold')
    axes[1, 2].set_ylabel('PET (mm/season)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Annual plots
    # Precipitation
    years_base = prec_ann_base.index.year
    years_stress = prec_ann_stress.index.year
    axes[2, 0].bar(years_base - 0.2, prec_ann_base['PREC'], width=0.4, label='Baseline', alpha=0.7)
    axes[2, 0].bar(years_stress + 0.2, prec_ann_stress['PREC'], width=0.4, label='Stress', alpha=0.7)
    axes[2, 0].set_title('Annual Precipitation', fontweight='bold')
    axes[2, 0].set_ylabel('Precipitation (mm/year)')
    axes[2, 0].set_xlabel('Year')
    axes[2, 0].legend()
    
    # Temperature
    axes[2, 1].bar(years_base - 0.2, ta_ann_base['TA'], width=0.4, label='Baseline', alpha=0.7)
    axes[2, 1].bar(years_stress + 0.2, ta_ann_stress['TA'], width=0.4, label='Stress', alpha=0.7)
    axes[2, 1].set_title('Annual Temperature', fontweight='bold')
    axes[2, 1].set_ylabel('Temperature (°C)')
    axes[2, 1].set_xlabel('Year')
    axes[2, 1].legend()
    
    # PET
    axes[2, 2].bar(years_base - 0.2, pet_ann_base['PET'], width=0.4, label='Baseline', alpha=0.7)
    axes[2, 2].bar(years_stress + 0.2, pet_ann_stress['PET'], width=0.4, label='Stress', alpha=0.7)
    axes[2, 2].set_title('Annual PET', fontweight='bold')
    axes[2, 2].set_ylabel('PET (mm/year)')
    axes[2, 2].set_xlabel('Year')
    axes[2, 2].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = figures_dir / f'{scenario_name}_duration{duration}_magnitude{magnitude}_station{station}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_path}")


def create_summary_statistics(base_data, stress_data, scenario_name, station, duration, magnitude):
    """Create summary statistics comparing base and stress scenarios"""
    
    (prec_mon_base, ta_mon_base, pet_mon_base,
     prec_seas_base, ta_seas_base, pet_seas_base,
     prec_ann_base, ta_ann_base, pet_ann_base) = base_data
    
    (prec_mon_stress, ta_mon_stress, pet_mon_stress,
     prec_seas_stress, ta_seas_stress, pet_seas_stress,
     prec_ann_stress, ta_ann_stress, pet_ann_stress) = stress_data
    
    summary = {
        'Scenario': scenario_name,
        'Station': station,
        'Duration': duration,
        'Magnitude': magnitude,
        
        # Annual averages
        'Prec_Annual_Base_Mean': prec_ann_base['PREC'].mean(),
        'Prec_Annual_Stress_Mean': prec_ann_stress['PREC'].mean(),
        'Prec_Annual_Change_%': ((prec_ann_stress['PREC'].mean() - prec_ann_base['PREC'].mean()) / 
                                  prec_ann_base['PREC'].mean() * 100),
        
        'TA_Annual_Base_Mean': ta_ann_base['TA'].mean(),
        'TA_Annual_Stress_Mean': ta_ann_stress['TA'].mean(),
        'TA_Annual_Change_°C': ta_ann_stress['TA'].mean() - ta_ann_base['TA'].mean(),
        
        'PET_Annual_Base_Mean': pet_ann_base['PET'].mean(),
        'PET_Annual_Stress_Mean': pet_ann_stress['PET'].mean(),
        'PET_Annual_Change_%': ((pet_ann_stress['PET'].mean() - pet_ann_base['PET'].mean()) / 
                                 pet_ann_base['PET'].mean() * 100),
    }
    
    return summary


def main():
    """Main analysis function"""
    
    base_path = Path(__file__).parent
    meteo_base_path = base_path / "input" / "2013-2023"
    stress_test_path = base_path / "input" / "stress_tests_meteo"
    figures_dir = base_path / "figures" / "stress_tests_meteo"
    
    # Create output directory
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    meteo_stations = [1443, 684, 1346, 2388, 259, 757, 1224]
    scenarios = ['spring-drought', 'summer-drought', 'spring-summer-drought']
    durations = [0, 2, 3]
    magnitudes = [0, 1, 2]
    
    all_summaries = []
    
    for station in meteo_stations:
        print(f"\n{'='*60}")
        print(f"Processing station {station}")
        print(f"{'='*60}")
        
        # Load baseline data
        try:
            prec_base, pet_base, ta_base = read_meteo(meteo_base_path / str(station))
            
            # Aggregate baseline data
            prec_mon_base, ta_mon_base, pet_mon_base = aggregate_to_monthly(prec_base, ta_base, pet_base)
            prec_seas_base, ta_seas_base, pet_seas_base = aggregate_to_seasonal(prec_base, ta_base, pet_base)
            prec_ann_base, ta_ann_base, pet_ann_base = aggregate_to_annual(prec_base, ta_base, pet_base)
            
            base_data = (prec_mon_base, ta_mon_base, pet_mon_base,
                        prec_seas_base, ta_seas_base, pet_seas_base,
                        prec_ann_base, ta_ann_base, pet_ann_base)
            
        except Exception as e:
            print(f"Error loading baseline data for station {station}: {e}")
            continue
        
        for scenario in scenarios:
            for duration in durations:
                for magnitude in magnitudes:
                    # Check which combinations exist based on the original script logic
                    if not ((magnitude == 0 and duration == 3) or
                            (magnitude == 1 and duration == 2) or
                            (magnitude == 2 and duration == 3) or
                            (magnitude == 2 and duration == 0)):
                        continue
                    
                    stress_path = stress_test_path / scenario / f"duration{duration}_magnitude{magnitude}" / str(station)
                    
                    if not os.path.exists(stress_path):
                        print(f"Path does not exist: {stress_path}")
                        continue
                    
                    try:
                        print(f"\nProcessing: {scenario} - Duration {duration} - Magnitude {magnitude}")
                        
                        # Load stress test data
                        prec_stress, pet_stress, ta_stress = read_meteo(stress_path)
                        
                        # Aggregate stress test data
                        prec_mon_stress, ta_mon_stress, pet_mon_stress = aggregate_to_monthly(prec_stress, ta_stress, pet_stress)
                        prec_seas_stress, ta_seas_stress, pet_seas_stress = aggregate_to_seasonal(prec_stress, ta_stress, pet_stress)
                        prec_ann_stress, ta_ann_stress, pet_ann_stress = aggregate_to_annual(prec_stress, ta_stress, pet_stress)
                        
                        stress_data = (prec_mon_stress, ta_mon_stress, pet_mon_stress,
                                      prec_seas_stress, ta_seas_stress, pet_seas_stress,
                                      prec_ann_stress, ta_ann_stress, pet_ann_stress)
                        
                        # Create plots
                        plot_comparisons(base_data, stress_data, scenario, station, duration, magnitude, figures_dir)
                        
                        # Create summary statistics
                        summary = create_summary_statistics(base_data, stress_data, scenario, station, duration, magnitude)
                        all_summaries.append(summary)
                        
                    except Exception as e:
                        print(f"Error processing {scenario} - Duration {duration} - Magnitude {magnitude}: {e}")
                        continue
        
        # Also process the wet scenario (no duration/magnitude variations)
        wet_scenario = 'spring-summer-wet'
        wet_path = stress_test_path / wet_scenario / str(station)
        
        if os.path.exists(wet_path):
            try:
                print(f"\nProcessing: {wet_scenario}")
                
                # Load wet scenario data
                prec_wet, pet_wet, ta_wet = read_meteo(wet_path)
                
                # Aggregate wet scenario data
                prec_mon_wet, ta_mon_wet, pet_mon_wet = aggregate_to_monthly(prec_wet, ta_wet, pet_wet)
                prec_seas_wet, ta_seas_wet, pet_seas_wet = aggregate_to_seasonal(prec_wet, ta_wet, pet_wet)
                prec_ann_wet, ta_ann_wet, pet_ann_wet = aggregate_to_annual(prec_wet, ta_wet, pet_wet)
                
                wet_data = (prec_mon_wet, ta_mon_wet, pet_mon_wet,
                           prec_seas_wet, ta_seas_wet, pet_seas_wet,
                           prec_ann_wet, ta_ann_wet, pet_ann_wet)
                
                # Create plots (using duration=0, magnitude=0 as placeholders)
                plot_comparisons(base_data, wet_data, wet_scenario, station, 0, 0, figures_dir)
                
                # Create summary statistics
                summary = create_summary_statistics(base_data, wet_data, wet_scenario, station, 0, 0)
                all_summaries.append(summary)
                
            except Exception as e:
                print(f"Error processing {wet_scenario}: {e}")
    
    # Save summary statistics to CSV
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = figures_dir / 'stress_test_summary_statistics.csv'
        summary_df.to_csv(summary_path, index=False, sep=';')
        print(f"\n{'='*60}")
        print(f"Summary statistics saved to: {summary_path}")
        print(f"{'='*60}")
        
        # Display summary
        print("\nSummary Statistics:")
        print(summary_df.to_string())


if __name__ == "__main__":
    main()