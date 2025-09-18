import sys
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import UEDGE_utils.analysis as ana
import UEDGE_utils.plot as utplt
from uedge import *
from uedge.hdf5 import *
from uedge.rundt import *
import uedge_mvu.plot as mp
import uedge_mvu.utils as mu
import uedge_mvu.analysis as mana
import uedge_mvu.tstep as ut
import UEDGE_utils.analysis as ana
from runcase import *
import traceback
import heat_code

setGrid()
setPhysics(impFrac=0,fluxLimit=True)
setDChi(kye=0.46, kyi=0.46, difni=0.5,nonuniform = True, kye_sol=0.26)
setBoundaryConditions(ncore=6.2e19, pcoree=2.0e6, pcorei=2.0e6, recycp=0.98, owall_puff=0)
setimpmodel(impmodel=True)


bbb.cion=3
bbb.oldseec=0
bbb.restart=1
bbb.nusp_imp = 3
bbb.icntnunk=0
bbb.kye=0.46
bbb.kyi = 0.46
setDChi(kye=0.36, kyi=0.36, difni=0.5,nonuniform = True, kye_sol=0.26)



t_run = 5 # s
dt_each = 5e-3




try:
    current_dir = os.getcwd()
    hdf5_dir = os.path.join(current_dir, "run_last_iterations")
    csv_dir = os.path.join(current_dir, "fngxrb_use")

    if not os.path.isdir(hdf5_dir):
        raise FileNotFoundError(f"Directory {hdf5_dir} not found.")

    # Find all run_last_{i}.hdf5 files and sort by i
    file_list = sorted([
        f for f in os.listdir(hdf5_dir)
        if f.startswith("run_last_") and f.endswith(".hdf5")
    ], key=lambda x: int(x.split("_")[-1].split(".")[0]))

    nx = len(file_list)
    print("Number of iterations found:", nx)

    if nx > 0:
        last_file = file_list[-1]  # e.g., run_last_15.hdf5
        iter_num = last_file.split("_")[-1].split(".")[0]
        hdf5_file_path = os.path.join(hdf5_dir, last_file)

        print(f"Loading HDF5 from: {hdf5_file_path}")
        hdf5_restore(hdf5_file_path)

        # Try loading corresponding CSV
        csv_file_name = f"fngxrb_use_{iter_num}.csv"
        csv_file_path = os.path.join(csv_dir, csv_file_name)

        if os.path.isfile(csv_file_path):
            print(f"Loading CSV from: {csv_file_path}")
            bbb.fngxrb_use[:, 1, 0] = np.loadtxt(csv_file_path, delimiter=',')
        else:
            print(f"CSV file {csv_file_path} not found. Skipping CSV data assignment.")
    else:
        print("No HDF5 iteration files found. Restoring from final.hdf5")
        hdf5_restore("./final.hdf5")

except Exception as e:
    print(f"Error encountered: {e}")
    print("Restoring from final.hdf5")
    hdf5_restore("./final.hdf5")


bbb.ftol = 1e-5
bbb.dtreal = 1e-10
bbb.issfon = 1
bbb.isbcwdt = 1
bbb.exmain()



def eval_Li_evap_at_T_Cel(temperature):
    a1 = 5.055  
    b1 = -8023.0
    xm1 = 6.939 
    tempK = temperature + 273.15

    if np.any(tempK <= 0):
        raise ValueError("Temperature must be above absolute zero (-273.15°C).")

    vpres1 = 760 * 10**(a1 + b1 / tempK)  

    sqrt_argument = xm1 * tempK
    if np.any(sqrt_argument <= 0):
        raise ValueError(f"Invalid value for sqrt: xm1 * tempK has non-positive values.")

    fluxEvap = 1e4 * 3.513e22 * vpres1 / np.sqrt(sqrt_argument)  # Evaporation flux
    return fluxEvap


def flux_Li_Phys_Sput(bbb, com, Yield=1e-3, UEDGE=False):
    if not UEDGE:
        kB = 1.3806e-23
        eV = 1.6022e-19
        fd = bbb.fnix[com.nx, :, 0] * np.cos(com.angfx[com.nx, :]) / com.sxnp[com.nx, :]
        ft = 0
        fli = 0 * (ft + fd) * 0.65
        ylid = Yield
        ylit = 0.001
        ylili = 0.3
        fneut = 0.35
        fneutAd = 1
        fluxPhysSput = fneut * (fd * ylid + ft * ylit + fli * ylili)
        return fluxPhysSput
    else:
        print('UEDGE model for Li physical sputtering')
        fluxPhysSput = bbb.sputflxrb[:, 1, 0] / com.sxnp[com.nx, :]
        return fluxPhysSput


def flux_Li_Ad_atom(final_temperature, bbb, com, Yield=1e-3, YpsYad=1, eneff=0.9, A=1e-7):
    yadoyps = YpsYad  # ratio of ad-atom to physical sputtering yield
    eneff = eneff  # effective energy (eV), 0.9
    aad = A  # constant
    ylid = Yield
    ylit = 0.001
    ft = 0
    kB = 1.3806e-23
    eV = 1.6022e-19
    tempK = final_temperature + 273.15
    fd = bbb.fnix[com.nx, :, 0] * np.cos(com.angfx[com.nx, :]) / com.sxnp[com.nx, :]
    fneutAd = 1
    fluxAd = fd * yadoyps / (1 + aad * np.exp(eV * eneff / (kB * tempK)))

    return fluxAd



def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

def save_csv(arr, directory, basename, i, delimiter=","):
    ensure_dir(directory)
    fname = os.path.join(directory, f"{basename}_{int(i)}.csv")
    np.savetxt(fname, arr, delimiter=delimiter)
    print(f"Saved {basename} in file: {fname}")

def save_npy(arr, filename):
    np.save(filename, arr)
    print(f"Saved {filename}")

def save_hdf5(savefile, directory="final_iterations"):
    ensure_dir(directory)
    full_path = os.path.join(directory, savefile)
    hdf5_save(full_path)
    print(f"Saved HDF5 file: {full_path}")

def save_png(plot_func, directory, basename, i, save=False):
    ensure_dir(directory)
    fname = os.path.join(directory, f"{basename}_{int(i)}.png")
    plot_func()  # This should create the plot, but not save it
    if save:
        plt.savefig(fname, dpi=300)
        print(f"Saved plot: {fname}")
    plt.close()

def save_surface_heatflux_components(ana, i, target):
    data = ana.get_surface_heatflux_components(target=target)
    keys = list(data.keys())
    arrs = [np.array(data[k_]).reshape(-1) for k_ in keys]
    # Ensure all arrays are the same length
    min_len = min(len(a) for a in arrs)
    arrs = [a[:min_len] for a in arrs]
    stacked = np.column_stack(arrs)
    outdir = "surface_heatflux_components"
    ensure_dir(outdir)
    fname = os.path.join(outdir, f"surface_heatflux_components_{target}_{int(i)}.csv")
    header = ",".join(keys)
    np.savetxt(fname, stacked, delimiter=",", header=header, comments='')
    print(f"Saved {fname}")

# --- Main Variables ---

Tsurf_max = []
phi_Li = []
qmax = []
it = []

Te_max_odiv = []
ne_max_odiv = []
zeff_OMP_sep = []
C_Li_omp_sep = []
ni_omp_sep = []
ne_omp_sep = []
Te_omp_sep = []
C_Li_all_sep_avg = []
phi_Li_source_odiv = []
phi_Li_source_idiv = []
phi_Li_source_pfr = []
phi_Li_source_wall = []
Li_rad = []
phi_Li_odiv = []
phi_Li_wall = []
phi_Li_idiv = []
pump_Li_odiv = []
pump_Li_wall = []
pump_Li_idiv = []
Li_ionization = []

# --- Main Loop Setup ---


n = int(t_run / dt_each)
i = 0
print("current is is:", i)

try:
    current_dir = os.getcwd()
    hdf5_dir = os.path.join(current_dir, "run_last_iterations")

    file_list = sorted([
        f for f in os.listdir(hdf5_dir)
        if f.startswith("run_last_") and f.endswith(".hdf5")
    ], key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if not file_list:
        i = 0
        print("No HDF5 files found, i set to 0")
    else:
        nx = len(file_list)
        i = nx
        print(f"Number of iterations found: {nx} and i updated as {i}")

except Exception as e:
    i = 0
    print(f"Error encountered: {e}")
    print("Setting i = 0 due to missing folder or read error")

while i <= n:
    bbb.pradpltwl()
    bbb.plateflux()
    q_rad = bbb.pwr_pltz[:, 1] + bbb.pwr_plth[:, 1]
    q_data = (bbb.sdrrb + bbb.sdtrb).reshape(-1)
    print('size of the data is :', len(q_data))
    print(type(bbb.sdrrb), type(bbb.sdtrb))
    print(bbb.sdrrb.shape, bbb.sdtrb.shape)
    q_data = np.round(q_data, 2)
    save_npy(q_data, 'q_data.npy')

    q_max2 = np.max(q_data)
    print('q_perp is done')
    print('Max q is :', q_max2)

    print('---Calling heat code----')
    try:
        T, no_it, temp_surf, Gamma, qsurf, qion = heat_code.run_heat_simulation(bbb=bbb, com=com, t_sim=dt_each)
        Tsurf = T 
    except Exception as e:
        print(f"Error executing heat_code.py: {e}")
        traceback.print_exc()
    

    thickness_threshold = 1e-6 
  
    np.save('T_surf2D.npy', T)
    T_max = np.max(Tsurf[:, 1])
    print('Peak temp is :', T_max)
    Tsurf_max.append(T_max)
    final_temperature = Tsurf[:, 1]  
    fluxEvap = eval_Li_evap_at_T_Cel(final_temperature)
    fluxPhysSput = flux_Li_Phys_Sput(bbb=bbb, com=com, UEDGE=True)
    fluxAd = flux_Li_Ad_atom(final_temperature, bbb=bbb, com=com, Yield=1e-3, YpsYad=1e-3, eneff=0.9, A=1e-7)
    tot = fluxEvap + fluxAd
    try:
       Li_thickness_read = np.load("Li_thickness_last.npy")
       print("Minimum Li thickness (m):", np.min(Li_thickness_read))
    except FileNotFoundError:
       Li_thickness_read = lithium_thickness  # fresh start

    tot = np.where(Li_thickness_read > thickness_threshold, tot, 0.0)

    save_csv(Tsurf[:, 1], "Tsurf_Li", "T_surfit", i)
    save_csv(qsurf, "q_Li_surface", "q_Li_surface", i)
    save_csv(qion, "q_ion", "q_ion", i)
    save_csv(Gamma, "Gamma_net", "Gamma_Li_surface", i)
    save_csv(fluxEvap, "Gamma_Li", "Evap_flux", i)
    save_csv(fluxPhysSput, "Gamma_Li", "PhysSput_flux", i)
    save_csv(fluxAd, "Gamma_Li", "Adstom_flux", i)
    save_csv(tot, "Gamma_Li", "Total_Li_flux", i)

    print('----Heat code completed and output obtained')
    print('Length of the surface temperature is:', len(Tsurf))
    print('---Heat code is done----')

    print("Total Li flux is :", len(tot))
    print("phi_Li^Odiv :", np.sum(tot * com.sxnp[com.nx, :]))
    print('-----Li upper limit sets to 1e22---')
    Li_int_flux = np.sum(tot * com.sxnp[com.nx, :])
    phi_Li.append(Li_int_flux)

    bbb.fngxrb_use[:, 1, 0] = tot * com.sxnp[com.nx, :]
    print('sum of evaporation and add atom is :', np.sum(bbb.fngxrb_use))

    save_csv(bbb.fngxrb_use[:, 1, 0], "fngxrb_use", "fngxrb_use", i)

    savefile = f"final_iteration_{int(i)}.hdf5"
    save_hdf5(savefile)

    Phy_sput_Li = np.sum(bbb.sputflxrb[:, 1, 0])
    print('Li sput phys is :', Phy_sput_Li)

    print("Completed heat code, now running UEDGE code with the updated Li flux")
    bbb.restart = 1
    bbb.itermx = 10
    bbb.dtreal = 1e-10
    bbb.ftol = 1e-5
    bbb.issfon = 1
    bbb.isbcwdt = 1
    bbb.exmain()

    ut.uestep(dt_each, depth_max=20, nrefine=3, reset=True)
  


    print("******done, now save data*****")
    
    if bbb.iterm == 1:
        save_npy(final_temperature, 'final.npy')
        bbb.pradpltwl()
        bbb.plateflux()
        q_rad = bbb.pwr_pltz[:, 1] + bbb.pwr_plth[:, 1]
        q_data = (bbb.sdrrb + bbb.sdtrb).reshape(-1)
        q_max = np.max(q_data)
        qmax.append(q_max)
        print("Max q_perp is :", q_max)
        save_csv(q_data, "q_perp", "q_perpit", i)
        savefile = f"run_last_{int(i)}.hdf5"
        save_hdf5(savefile, directory="run_last_iterations")
        print("Saving lasted out in file: ", savefile)

        Te_max = np.max(bbb.te[com.nx, :] / bbb.ev)
        ne_max = np.max(bbb.ne[com.nx, :])
        Te_max_odiv.append(Te_max)
        ne_max_odiv.append(ne_max)
        zeff_omp = bbb.zeff[bbb.ixmp, com.iysptrx+1]
        zeff_OMP_sep.append(zeff_omp)
        n_Li = (bbb.ni[bbb.ixmp, com.iysptrx+1, com.nhsp] +
                bbb.ni[bbb.ixmp, com.iysptrx+1, com.nhsp+1] +
                bbb.ni[bbb.ixmp, com.iysptrx+1, com.nhsp+2])
        C_Li_OMP = n_Li / bbb.ne[bbb.ixmp, com.iysptrx+1]
        C_Li_omp_sep.append(C_Li_OMP)
        ni_omp = bbb.ni[bbb.ixmp, com.iysptrx+1, 0]
        ni_omp_sep.append(ni_omp)
        ne_omp = bbb.ne[bbb.ixmp, com.iysptrx+1]
        ne_omp_sep.append(ne_omp)
        Te_omp = bbb.te[bbb.ixmp, com.iysptrx+1] / bbb.ev
        Te_omp_sep.append(Te_omp)
        save_csv(bbb.ni[:, :, 2], "n_Li1", "n_Li1", i)
        save_csv(bbb.ng[:, :, 1], "n_atom", "n_0", i)
        save_csv(bbb.fnix[com.nx, :, 0], "Phi_D1_odiv", "Phi_D1", i)
        save_csv(bbb.fnix[com.nx, :, com.nhsp], "Phi_Li1_odiv", "Phi_Li1", i)
        save_csv(bbb.fnix[com.nx, :, com.nhsp+1], "Phi_Li2_odiv", "Phi_Li2", i)
        save_csv(bbb.fnix[com.nx, :, com.nhsp+2], "Phi_Li3_odiv", "Phi_Li3", i)
        save_csv(bbb.fniy[:, com.ny, com.nhsp+2], "Phi_Li3_wall", "Phi_Li3", i)
        save_csv(bbb.te / bbb.ev, "T_e", "T_e", i)
        save_csv(bbb.ni[:, :, com.nhsp+1], "n_Li2", "n_Li2", i)
        save_csv(bbb.ni[:, :, com.nhsp+2], "n_Li3", "n_Li3", i)
        ensure_dir("n_e")
        np.save(os.path.join("n_e", f"n_e_{int(i)}"), bbb.ne)
        n_Li_all_sep = (bbb.ni[:, com.iysptrx, com.nhsp] +
                        bbb.ni[:, com.iysptrx, com.nhsp+1] +
                        bbb.ni[:, com.iysptrx, com.nhsp+2])
        ne_all_sep = bbb.ne[:, com.iysptrx]
        C_Li_all_sep = n_Li_all_sep / ne_all_sep
        save_csv(C_Li_all_sep, "C_Li", "C_Li_sep_all", i)
        C_Li_all_sep_avg.append(np.average(C_Li_all_sep))

        n_Li_rad = (bbb.ni[bbb.ixmp, :, com.nhsp] +
                    bbb.ni[bbb.ixmp, :, com.nhsp+1] +
                    bbb.ni[bbb.ixmp, :, com.nhsp+2])
        C_Li_OMP_rad = n_Li / bbb.ne[bbb.ixmp, :]
        zeff_omp_rad = bbb.zeff[bbb.ixmp, :]
        save_csv(C_Li_OMP_rad, "C_Li_omp", "CLi_prof", i)
        save_csv(zeff_omp_rad, "Z_eff_omp", "Zeff_prof", i)
        save_csv(bbb.prad[:, :], "Li_rad", "Li_rad", i)
        phi_Li_source_odiv.append(np.sum(bbb.sputflxrb))
        phi_Li_source_idiv.append(np.sum(bbb.sputflxlb))
        phi_Li_source_pfr.append(np.sum(bbb.sputflxpf))
        phi_Li_source_wall.append(np.sum(bbb.sputflxw))
        Li_rad.append(np.sum(bbb.prad[:, :] * com.vol))
        phi_Li_odiv.append(np.sum(bbb.fnix[com.nx, :, com.nhsp:com.nhsp+3]))
        phi_Li_idiv.append(np.sum(np.abs(bbb.fnix[0, :, com.nhsp:com.nhsp+3])))
        phi_Li_wall.append(np.sum(np.abs(bbb.fniy[:, com.ny, com.nhsp:com.nhsp+3])))
        pump_Li_odiv.append(np.sum((1 - bbb.recycp[1]) * bbb.fnix[com.nx, :, com.nhsp:com.nhsp+3]))
        pump_Li_idiv.append(np.sum((1 - bbb.recycp[1]) * bbb.fnix[0, :, com.nhsp:com.nhsp+3]))
        pump_Li_wall.append(np.sum((1 - bbb.recycw[1]) * bbb.fniy[:, com.ny, com.nhsp:com.nhsp+3]))
        Li_ionization.append(np.sum(np.abs(bbb.psor[:, :, com.nhsp:com.nhsp+3])))

        # --- Save surface heatflux components and plots for both targets ---
        for target in ['outer', 'inner']:
            save_surface_heatflux_components(ana, i, target)
            save_png(
      		  lambda: utplt.plot_surface_heatflux_components(target=target),
      		  "q_surf_output_png",
       		  f"q_surf_{target}",
       				 i,
      			  save=True)
   				 
    it.append(i)
    i += 1
    print("Iteration:", i)


qmax = np.array(qmax)
tsurf = np.array(Tsurf_max)
Phi_Li = np.array(phi_Li)

savefile = "final_iteration.hdf5"
save_hdf5(savefile)

np.savetxt("qmax.csv", qmax, delimiter=",")
np.savetxt("Tsurf.csv", tsurf, delimiter=",")
np.savetxt("It.csv", it, delimiter=",")
np.savetxt("Phi_Li.csv", Phi_Li, delimiter=",")

data = {
    "phi_Li_source_odiv": phi_Li_source_odiv,
    "phi_Li_source_idiv": phi_Li_source_idiv,
    "phi_Li_source_pfr": phi_Li_source_pfr,
    "phi_Li_source_wall": phi_Li_source_wall,
    "Li_rad": Li_rad,
    "phi_Li_odiv": phi_Li_odiv,
    "phi_Li_wall": phi_Li_wall,
    "phi_Li_idiv": phi_Li_idiv,
    "pump_Li_odiv": pump_Li_odiv,
    "pump_Li_wall": pump_Li_wall,
    "pump_Li_idiv": pump_Li_idiv,
    "Li_ionization": Li_ionization,
    "Te_max_odiv": Te_max_odiv,
    "ne_max_odiv": ne_max_odiv,
    "zeff_omp_sep": zeff_OMP_sep,
    "C_Li_omp_sep": C_Li_omp_sep,
    "ni_omp_sep": ni_omp_sep,
    "Te_omp_sep": Te_omp_sep,
    "ne_omp_sep": ne_omp_sep,
    "C_Li_sep_all": C_Li_all_sep_avg,
}
df = pd.DataFrame(data)
csv_filename = "Li_all.csv"
df.to_csv(csv_filename, index=False)
print(f"Data successfully saved to {csv_filename}")
