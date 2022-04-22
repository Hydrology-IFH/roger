# y-axis labels
_Y_LABS_DAILY = {'prec': r'$PREC$ [mm $day^{-1}$]',
                 'aet': r'$ET$ [mm $day^{-1}$]',
                 'q_ss': r'$PERC$ [mm $day^{-1}$]',
                 'dS': r'$\Delta$$S$ [mm $day^{-1}$]',
                 'S': r'$S$ [mm]',
                 'theta': r'$\theta$ [-]',
                 'theta_rz': r'$\theta_{roots}$ [-]',
                 'theta_ss': r'$\theta_{subsoil}$ [-]',
                }

_Y_LABS_CUM = {'prec': r'$PREC$ [mm]',
             'aet': r'$ET$ [mm]',
             'q_ss': r'$PERC$ [mm]',
             'dS': r'$\Delta$$S$ [mm]',
             'S': r'$S$ [mm]',
             'theta': r'$\theta$ [-]',
             'theta_rz': r'$\theta_{roots}$ [-]',
             'theta_ss': r'$\theta_{subsoil}$ [-]',
            }

# long names of variables used for netcdf
_LONG_NAME = {'prec': r'Precipitation',
             'pet': r'Potential evapotranspiration',
             'ta': r'Air temperature',
             'Cl': 'Chloride in precipitation',
             'Br': 'Bromide in injection',
             'd2H': 'Deuterium in precipitation',
             'd18O': 'Oxygen-18 in precipitation',
             'Norg': 'Organic nitrogen fertilizer',
             'Nmin': 'Mineral nitrogen fertilizer',
            }

# units of variables used for netcdf
_UNITS = {'prec': 'mm/dt',
             'pet': 'mm/dt',
             'ta': 'dgC',
             'Cl': 'mg/l',
             'Br': 'mg',
             'd2H': 'permil',
             'd18O': 'permil',
             'Norg': 'kg/ha',
             'Nmin': 'kg/ha',
            }
