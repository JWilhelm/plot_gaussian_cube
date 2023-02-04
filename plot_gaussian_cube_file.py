import numpy as np
import matplotlib.pyplot as plt
import math

def plot_gaussian_cube(file_name, png_name, z_min_A, z_max_A):

    # Load the data from the Gaussian cube file
    with open(file_name, 'r') as f:
        lines     = f.readlines()
        header    = lines[2:6]
        n_atoms   = int(header[0].split()[0])
        n_x       = int(header[1].split()[0])
        n_y       = int(header[2].split()[0])
        n_z       = int(header[3].split()[0])
        origin    = np.zeros(3)
        origin[:] = [float(x) for x in header[0].split()[1:]]
        vec_a     = np.zeros(3)
        vec_b     = np.zeros(3)
        vec_c     = np.zeros(3)
        vec_a[:]  = [float(x) for x in header[1].split()[1:]]
        vec_b[:]  = [float(x) for x in header[2].split()[1:]]
        vec_c[:]  = [float(x) for x in header[3].split()[1:]]
        if abs(vec_c[0]) > 1.0E-10 or abs(vec_c[1]) > 1.0E-10: 
           print("Script assumes that third vector is aligned in z, exit")
           quit()
        atom_data = [list(map(float, line.split())) for line in lines[6:6+n_atoms]]
        atom_type = np.zeros(n_atoms)
        atom_coord  = np.zeros([n_atoms, 3])
        for i_atom in range(n_atoms):
            atom_type[i_atom]   = atom_data[i_atom][0]
            atom_coord [i_atom,:] = atom_data[i_atom][2:]
        print("first atom =", atom_type[0],  atom_coord[0,:])
        print("last atom  =", atom_type[-1], atom_coord[-1,:])
        n_a = int(header[1].split()[0])
        n_b = int(header[2].split()[0])
        n_c = int(header[3].split()[0])
        data = [list(map(float, line.split())) for line in lines[6+n_atoms:]]
    data_3d = np.zeros([n_a, n_b, n_c])

    row_index = 0
    col_index = 0
    do_exit = False
    for i_a in range(n_a):
      for j_b in range(n_b):
        for k_c in range(n_c):

          data_3d[i_a,j_b,k_c] = data[row_index][col_index]

          if len(data[row_index]) == col_index+1:
            row_index = row_index + 1
            col_index = 0
          else:
            col_index = col_index + 1

          if len(data) == row_index:
              do_exit = True

          if do_exit: 
            break
        if do_exit:
          break
      if do_exit:
        break

    n_x_y_max = max( n_x , n_y)

    n_2d = n_x_y_max

    data_2d = np.zeros( [ n_2d, n_2d ] )
    longest_axis_basis_vectors = max( abs(vec_a[0]+vec_b[0]) , abs(vec_a[0]-vec_b[0]) , abs(vec_a[1]+vec_b[1]) , abs(vec_a[1]-vec_b[1]) )

    print("longest_axis_basis_vectors =", longest_axis_basis_vectors)
    print("Plotting area is square of size =", n_2d*longest_axis_basis_vectors)

    hmat_cell = np.zeros([2,2])
    hmat_cell[0,0] = vec_a[0]
    hmat_cell[1,0] = vec_a[1]
    hmat_cell[0,1] = vec_b[0]
    hmat_cell[1,1] = vec_b[1]
    vec_r = np.zeros(2)

    atom_coord_plotting = np.zeros([n_atoms, 2])
    for i_atom in range(n_atoms):
        atom_coord_plotting[i_atom,:] = atom_coord[i_atom,0:2] / longest_axis_basis_vectors
        print(i_atom, atom_coord[i_atom,0:2], atom_coord_plotting[i_atom,:])

    hmat_cell_inv = np.linalg.inv(hmat_cell)
    expansion_coeff = np.zeros(2)

    # Determine the indices of the xy plane with height z
    for z_index in range(n_z):

       actual_z_value_in_A = z_index*vec_c[2]*0.529
       if actual_z_value_in_A < z_min_A: continue
       if actual_z_value_in_A > z_max_A: continue

       for i_x in range(n_2d):
         for j_y in range(n_2d):
           vec_r[0] = i_x*longest_axis_basis_vectors
           vec_r[1] = j_y*longest_axis_basis_vectors
           expansion_coeff = np.matmul(hmat_cell_inv, vec_r)
           expansion_coeff[0] = expansion_coeff[0]
           expansion_coeff[1] = expansion_coeff[1]
           exp_coeff_x_lesser = math.floor(expansion_coeff[0])
           exp_coeff_x_upper  = exp_coeff_x_lesser + 1
           exp_coeff_y_lesser = math.floor(expansion_coeff[1])
           exp_coeff_y_upper  = exp_coeff_y_lesser + 1
           i_x_lesser = exp_coeff_x_lesser % n_x
           i_x_upper  = exp_coeff_x_upper  % n_x
           j_y_lesser = exp_coeff_y_lesser % n_y
           j_y_upper  = exp_coeff_y_upper  % n_y
           # weights
           w_x_lesser = exp_coeff_x_upper - expansion_coeff[0]
           w_x_upper  = 1.0 - w_x_lesser
           w_y_lesser = exp_coeff_y_upper - expansion_coeff[1]
           w_y_upper  = 1.0 - w_y_lesser

#           data_2d[i_x, j_y] = w_x_lesser * w_y_lesser * data_3d[i_x_lesser, j_y_lesser, z_index] + \
           data_2d[j_y, i_x] = w_x_lesser * w_y_lesser * data_3d[i_x_lesser, j_y_lesser, z_index] + \
                               w_x_upper  * w_y_lesser * data_3d[i_x_upper , j_y_lesser, z_index] + \
                               w_x_lesser * w_y_upper  * data_3d[i_x_lesser, j_y_upper , z_index] + \
                               w_x_upper  * w_y_upper  * data_3d[i_x_upper , j_y_upper , z_index]

       plt.clf()
       plt.xlabel("$x$ ($\mathrm{\AA}$)")
       plt.ylabel("$y$ ($\mathrm{\AA}$)")
       plt.gca().invert_yaxis()
       x_y_max      = n_2d*longest_axis_basis_vectors
       x_y_max_in_A = x_y_max*0.529
       plt.imshow(data_2d[::-1,:], extent=(0,x_y_max_in_A,0,x_y_max_in_A))
       plt.colorbar()
       plt.savefig(png_name+"_z_"+"{:.2f}".format(actual_z_value_in_A)+".png")

       # Map numbers to colors using a colormap
       cmap = plt.get_cmap('Reds')
       colors = [cmap(n/100) for n in atom_type]

       print("n_2d =", n_2d)
       print("x_y_max =", x_y_max)

       # Add the crosses
       cell_vec_a = vec_a*n_a
       cell_vec_b = vec_b*n_b
       for i_atom, color in enumerate(colors):
         for a_neighbor_cell in [-2,-1,0,1,2]:
           for b_neighbor_cell in [-2,-1,0,1,2]:
             atom_coord_cell = atom_coord[i_atom, :] + a_neighbor_cell*cell_vec_a + b_neighbor_cell*cell_vec_b
             if atom_coord_cell[0] > 0 and atom_coord_cell[0] < x_y_max and atom_coord_cell[1] > 0 and atom_coord_cell[1] < x_y_max:
                print("atom coord in Angström: ", atom_coord_cell[0]*0.529, atom_coord_cell[1]*0.529)
#                plt.plot(atom_coord_cell[0]/longest_axis_basis_vectors, atom_coord_cell[1]/longest_axis_basis_vectors, marker='x', markersize=6, markeredgewidth=2, color=color)
#                plt.plot(atom_coord_cell[0]/x_y_max*n_2d, atom_coord_cell[1]/x_y_max*n_2d, marker='x', markersize=6, markeredgewidth=2, color=color)
                plt.plot(atom_coord_cell[0]*0.529, atom_coord_cell[1]*0.529, marker='x', markersize=6, markeredgewidth=2, color=color)

       x_pts = np.zeros(2)
       y_pts = np.zeros(2)
#       for a_neighbor_cell in [-2,-1,0,1,2]:
#         for b_neighbor_cell in [-2,-1,0,1,2]:
       for a_neighbor_cell in [-2.25,-1.25,-0.25,0.75,1.75]:
         for b_neighbor_cell in [-2.5,-1.5,0.5,1.5,2.5]:
            x_pts[0] = a_neighbor_cell*cell_vec_a[0] + b_neighbor_cell*cell_vec_b[0] - 1000*vec_a[0]
            x_pts[1] = a_neighbor_cell*cell_vec_a[0] + b_neighbor_cell*cell_vec_b[0] + 1000*vec_a[0]
            y_pts[0] = a_neighbor_cell*cell_vec_a[1] + b_neighbor_cell*cell_vec_b[1] - 1000*vec_a[1]
            y_pts[1] = a_neighbor_cell*cell_vec_a[1] + b_neighbor_cell*cell_vec_b[1] + 1000*vec_a[1]
            plt.plot(x_pts*0.529, y_pts*0.529, '-w', linewidth=2)
            x_pts[0] = a_neighbor_cell*cell_vec_a[0] + b_neighbor_cell*cell_vec_b[0] - 1000*vec_b[0]
            x_pts[1] = a_neighbor_cell*cell_vec_a[0] + b_neighbor_cell*cell_vec_b[0] + 1000*vec_b[0]
            y_pts[0] = a_neighbor_cell*cell_vec_a[1] + b_neighbor_cell*cell_vec_b[1] - 1000*vec_b[1]
            y_pts[1] = a_neighbor_cell*cell_vec_a[1] + b_neighbor_cell*cell_vec_b[1] + 1000*vec_b[1]
            plt.plot(x_pts*0.529, y_pts*0.529, '-w', linewidth=2)

       plt.xlim(0,x_y_max_in_A)
       plt.ylim(0,x_y_max_in_A)

       plt.savefig(png_name+"_z_"+"{:.2f}".format(actual_z_value_in_A)+"_with_atom_positions.png")

## z in Anström
#for z in np.arange(15):

#plot_gaussian_cube("MoS2-LOCAL_BANDGAP-DFT_VBM_in_eV.cube", "DFT_VBM_2d", 4.0, 6.0)
#plot_gaussian_cube("MoS2-LOCAL_BANDGAP-DFT_CBM_in_eV.cube", "DFT_CBM_2d", 4.0, 6.0)
#plot_gaussian_cube("MoS2-LOCAL_BANDGAP-DFT_Gap_in_eV.cube", "DFT_Gap_2d", 4.0, 6.0)
#plot_gaussian_cube("MoS2-LOCAL_BANDGAP-GW_Gap_in_eV.cube", "GW_Gap_2d", 4.0, 6.0)
#plot_gaussian_cube("MoS2-LOCAL_BANDGAP-GW_VBM_in_eV.cube", "GW_VBM_2d", 4.0, 6.0)
#plot_gaussian_cube("MoS2-LOCAL_BANDGAP-GW_CBM_in_eV.cube", "GW_CBM_2d", 4.0, 6.0)
#plot_gaussian_cube("MoS2-LOCAL_BANDGAP-GW_LDOS_VBM_in_eV.cube", "GW_LDOS_VBM_2d", 4.0, 6.0)
#plot_gaussian_cube("MoS2-LOCAL_BANDGAP-GW_LDOS_CBM_in_eV.cube", "GW_LDOS_CBM_2d", 4.0, 6.0)
#plot_gaussian_cube("MoS2-LOCAL_BANDGAP-DFT_LDOS_VBM_in_eV.cube", "DFT_LDOS_VBM_2d", 4.0, 6.0)
#plot_gaussian_cube("MoS2-LOCAL_BANDGAP-DFT_LDOS_CBM_in_eV.cube", "DFT_LDOS_CBM_2d", 4.0, 6.0)
plot_gaussian_cube("MoS2-ELECTRON_DENSITY-1_0.cube", "E-Density", 4.0, 6.0)
