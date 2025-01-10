import os
# Content format for the mock chi^2 files, including parameters as per the provided structure
# Define the directory where files will be created
# Chi^2 values for each file
chi_squared_values = [286.68, 289.12, 295.95, 306.40, 280.55, 286.20, 285.80, 387.30, 386.10, 185.65, 285.65,205.65,215.65,230.0,185.5]
# Parameters format (random example for each file)
parameters_content = """
#        omega_cdm,     ln10^{10}A_s,          omega_b,              n_s,                h, parameters_smg__2, parameters_smg__3,  screening_scale,             A_IA,            c_min,             D_z1,             D_z2,             D_z3,             D_z4,             D_z5
 1.198000e-01    2.931423e+00    2.387000e-02    9.375600e-01    6.671800e-01    2.000000e-01    2.077000e+00   -4.479000e-01    2.533000e+00    2.256789e+00   -1.264985e-03    1.803575e-01    1.110001e+00    1.379634e+00   -1.266370e+00
"""
# Create 10 files with different chi^2 values and the parameters content
for i, chi2 in enumerate(chi_squared_values):
    file_name = f"KiDSxBOSS_mock_chi2_{i}.txt"
    directory = '/Users/matteograsso/Desktop/'
    file_path = os.path.join(directory, file_name)
    
    # Write the content to the file
    with open(file_path, 'w') as file:
        file.write(f"# minimized \\chi^2 = {chi2}\n")
        file.write(parameters_content)

directory
