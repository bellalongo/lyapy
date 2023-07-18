import matplotlib.pyplot as plt
import numpy as np
import copy
import emcee
import time
import joblib
from lyapy import *
from corner import corner

plt.ion()

plt.rc('font', family='sans-serif', size=14)


"""
    Calculates the log-prior probability of theta
"""
def lnprior(theta, minmax, prior_Gauss, prior_list):
    priors = 0
    # Iterate through theta
    for i in range(len(theta)):
        # Check if the parameters is out of range
        if (theta[i] < minmax[i][0]) or (theta[i] > minmax[i][1]):
            return -np.inf
        # Check if Guassian prior is specified
        if prior_Gauss[i]:
            priors += -0.5*((theta[i]-prior_list[i][0])/prior_list[i][1])**2
        # Check for doppler broadening parameter
        elif theta[i] == 'h1_b' and not prior_Gauss[i]:
            priors += np.log(theta[i])
        else:
            priors += 0 
    
    return priors


"""
    Calculates the log liklihood of a given model compared to observed data
"""
def lnlike(theta, x, y, yerr, resolution, variables, model_function, mask_data, xerr=None,debug=False):
    # Create masked arrays
    x = np.ma.array(x, mask=mask_data)
    y = np.ma.array(y, mask=mask_data)
    yerr = np.ma.array(yerr, mask=mask_data)

    y_model = model_function(x, resolution, theta, variables, lnlike=True)

    lnlike = 0

    for i in range(len(x)):
        log_xerr = np.log(2.*np.pi*xerr[i]**2) if xerr is not None else 0

        if yerr[i] is None:

            lnlike += -(np.sum( (y[i] - y_model[i])**2 )) / len(x[i])

        else:

            lnlike += -0.5*(np.sum( (y[i] - y_model[i])**2/yerr[i]**2 + np.log(2*np.pi*yerr[i]**2) + log_xerr)) 


    if debug:
        plt.figure()
        plt.plot(x,y,color='k')
        plt.plot(x,y_model,color='deeppink')
        print(lnlike)

        
    return lnlike

def lnprob(theta, x, y, yerr, resolution, variables, variables_order, model_function, mask_data, xerr=None, debug=False):

    theta_all = []
    range_all = []
    prior_Gauss = []
    prior_list = []
    i = 0
    for p in variables_order:

        range_all.append( [variables[p]['min'],variables[p]['max']] )

        if variables[p]['vary']:

            theta_all.append(theta[i])

            i = i+1

        else:

            theta_all.append(variables[p]['value'])

        if variables[p]['Gaussian prior']:

            prior_Gauss.append(True)

            prior_list.append([variables[p]['prior mean'],variables[p]['prior stddev']])

        else:

            prior_Gauss.append(False)
            
            prior_list.append(0)
                
    assert (i) == len(theta)


    lp = lnprior(theta_all, range_all, prior_Gauss, prior_list)
    
    if not np.isfinite(lp):

        return -np.inf

    #ll = lnlike_asymm(theta_all, x, y, yerr, resolution, variables, model_function, mask_data, xerr, debug)
    ll = lnlike(theta_all, x, y, yerr, resolution, variables, model_function, mask_data, xerr, debug)
    
    return lp + ll





def profile_plot(x, y, yerr, resolution, samples, model_function, variables, variables_order, perform_error=True, thin_out=1.0, plot_style = 'step', xerr=None, convolve=True):
    # Get the number of subplots
    number_of_subplots = len(x)

    if number_of_subplots > 5:
        # Create a layout of 2 rows and 1 column
        fig, axes = plt.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios':[3,1]})
    else:
        # Create a layout of 2 rows, and the number of subplots as the number of columns
        fig, axes = plt.subplots(2, number_of_subplots, sharex=False, figsize=(8*number_of_subplots,5), gridspec_kw={'height_ratios':[3,1]})

    # Make sure that there are samplers
    assert samples is not None, "%r needs to not be None" % samples

    # Extract the best fit parameters and store them
    parameters_best = []
    for variable_name in variables_order:
        if variables[variable_name]['vary']:
            parameters_best.append(variables[variable_name]['best'][2])
        else:
            parameters_best.append(variables[variable_name]['value'])

    output = model_function(x, resolution, parameters_best, variables, lnlike=False, convolve=convolve)
    assert isinstance(output, list), "%r is not a list" % output

    array_length = int(len(samples)/thin_out)
    number_of_arrays_per_line_to_store = len(output)
    number_of_lines = len(output[0])

    # Initialize the necessary dictionaries
    line_arrays_to_store_dic = {}
    line_percentiles_to_store_dic = {}
    reconstructed_fluxes_dic = {}

    # If there is a performance error
    if perform_error:
        # Reformat
        for line_number in range(number_of_lines):
            line_key_name = "Line{0}".format(line_number)
            line_arrays_to_store_dic[line_key_name] = np.zeros( (array_length, len(x[line_number]), \
                                number_of_arrays_per_line_to_store) )
            line_percentiles_to_store_dic[line_key_name] = \
                        np.zeros((number_of_arrays_per_line_to_store, len(x[line_number]), 5))
            reconstructed_fluxes_dic[line_key_name] = np.zeros( (array_length, \
                        number_of_arrays_per_line_to_store))
        index = -1

        # Iterate through the samples
        for i, sample in enumerate(samples):
            # Check for thin out
            if (i % thin_out) == 0:
                index += 1
                theta_all = []
                theta = []
                j=0

                # Initialize necessaru variables
                for variable_name in variables_order:
                    if variables[variable_name]['vary']:
                        theta_all.append(sample[j])
                        theta.append(sample[j])
                            
                        j += 1
                    else:
                        theta_all.append(variables[variable_name]['value'])
                
                # Compute the model output based on the calculated parameters (above)
                output = model_function(x, resolution, theta_all, variables, lnlike=False, convolve=convolve)

                # Reformat (same as above)
                for line_number in range(number_of_lines):
                    line_key_name = "Line{0}".format(line_number)
                    for array_number in range(number_of_arrays_per_line_to_store):
                        line_arrays_to_store_dic[line_key_name][index, :, array_number] = \
                                    output[array_number][line_number]

        for line_number in range(number_of_lines):
            line_key_name = "Line{0}".format(line_number)
            for array_number in range(number_of_arrays_per_line_to_store):
                for wave_index in range(len(x[line_number])):
                    line_percentiles_to_store_dic[line_key_name][array_number, wave_index, :] = \
                            np.percentile(line_arrays_to_store_dic[line_key_name][:, wave_index, array_number], \
                            [2.5, 15.9, 50., 84.1, 97.5])
                for sample_index in range(array_length):
                    reconstructed_fluxes_dic[line_key_name][sample_index, array_number] = np.trapz( \
                            line_arrays_to_store_dic[line_key_name][sample_index, :, array_number], \
                            x[line_number])

    # Compute the best parameters
    y_model_best = model_function( x, resolution, parameters_best, variables, lnlike=True)

    # Iterate through the subplots
    for i in range(number_of_subplots):
            # Plot the subplot data
            ax = axes if number_of_subplots == 1 else axes[:,i]
            if plot_style == 'step':
                ax[0].step( x[i], y[i], color='k', where='mid')
            elif plot_style == 'scatter':
                ax[0].plot( x[i], y[i], color='k', marker='o', linestyle='none')

            xerr_val = None if xerr is None else xerr[i]

            ax[0].errorbar( x[i], y[i], yerr= yerr[i], xerr=xerr_val, 
                         fmt="none",ecolor='limegreen',elinewidth=1,capthick=1)

            line_key_name = "Line{0}".format(i)

            if perform_error: 
                ax[0].fill_between( x[i], line_percentiles_to_store_dic[line_key_name][0, :, 0], 
                                   line_percentiles_to_store_dic[line_key_name][0, :, 4],
                                   color='lightgrey')
                ax[0].fill_between( x[i], line_percentiles_to_store_dic[line_key_name][0, :, 1], 
                                   line_percentiles_to_store_dic[line_key_name][0, :, 3],
                                   color='grey')
                ax[0].plot( x[i], line_percentiles_to_store_dic[line_key_name][0, :, 2], 
                                       color='deeppink',linewidth=1.5, linestyle='--')
            ax[0].plot( x[i], y_model_best[i], color='deeppink', linewidth=1.5)

            ax[0].minorticks_on()

            # Plot the residuals
            residuals = (y[i] - y_model_best[i])/yerr[i]

            ax[1].plot(x[i], residuals, 'ko')
            ax[1].hlines(0,np.min(x[i]),np.max(x[i]),color='grey',linestyle='--')
            ax[1].hlines(1,np.min(x[i]),np.max(x[i]),color='grey',linestyle=':')
            ax[1].hlines(-1,np.min(x[i]),np.max(x[i]),color='grey',linestyle=':')

            ax[1].minorticks_on()

            ax[0].ticklabel_format(useOffset=False)
            ax[1].ticklabel_format(useOffset=False)

    return line_percentiles_to_store_dic, reconstructed_fluxes_dic




def setup_sampler(x, y, yerr, resolution, nwalkers, variables, variables_order, my_model, mask_data, start_uniform=True, xerr=None,debug=False, lnprob=lnprob):
    # List of parameters being varyed this run
    varyparams = []
    # Set up variables that will be varied
    theta, scale, mins, maxs = [], [], [], []
    
    number_free_parameters  = 0
    # Identify which parameters need to be varied
    for p in variables_order: 
        if variables[p]['vary']:
            varyparams.append(p)
            theta.append(variables[p]['value'])
            scale.append(variables[p]['scale'])
            mins.append(variables[p]['min'])
            maxs.append(variables[p]['max'])
            number_free_parameters += 1
        # If the parameter is fixed, just record the starting value as the best value
        else:
            variables[p]['best'][0] = variables[p]['value']  
            
    # Print the parameters being used and their initial values
    print("Varying: ", varyparams)
    for p in variables.keys():
        print(p, variables[p]['value'])
    ndim = len(varyparams)

    # If the intial positions of the walkers should be evenly distributed
    if start_uniform: 
        pos0 = [np.random.uniform(low=mins, high=maxs) for i in range(nwalkers)]
    # Or initalized near the initial parameter values with Gaussian noise
    else:
        pos0 = [theta + scale*np.random.randn(ndim) for i in range (nwalkers)]

    # Create the sampler for the liklihood distribution
    sampler = emcee.EnsembleSampler(nwalkers, number_free_parameters, lnprob, 
            args=(x,y,yerr, resolution, variables, variables_order, my_model, mask_data, xerr,debug))

    return sampler, pos0


def perform_mcmc(sampler, pos0, nsteps, nruns=1,fresh_start=True):

    for i in range(nruns):

        print("Run #" + str(i + 1))

        start_time = time.time()

        if fresh_start & (i==0):

            print("Fresh start")

            pos, lnprob_vals, rstate = sampler.run_mcmc(pos0, nsteps, rstate0=np.random.get_state())

        else:

            print("picking up where the MCMC left off")

            with open("state.pkl", "rb") as f:

                pos0, lnprob_vals, rstate = joblib.load(f)

            pos, lnprob_vals, rstate = sampler.run_mcmc(pos0, nsteps, rstate0=rstate)

        end_time = time.time()

        print("Done.")
        print("Time to complete MCMC #"+ str(i+1) + ": {0:.1f} seconds"
                .format(end_time-start_time))

        
        save_sampler_state(pos, lnprob, rstate)

        save_sampler_chain(sampler, fresh_start, i)

        sampler_chain = get_sampler_chain()

    return sampler_chain

def save_sampler_state(pos, lnprob, rstate):

    with open("state.pkl", "wb") as f:

        joblib.dump([pos, lnprob, rstate], f)

    return

def save_sampler_chain(sampler, fresh_start, i):

    if fresh_start & (i==0):

        print("Saving option 1")

        with open("chain.pkl", "wb") as f:

            joblib.dump(sampler.chain, f)

    else:

        print("Saving option 2")

        with open("chain.pkl", "rb") as f:

            old_sample_chain = joblib.load(f)

        sampler_chain_array = np.concatenate((old_sample_chain, sampler.chain), axis=1)

        with open("chain.pkl", "wb") as f:

            joblib.dump(sampler_chain_array, f)

    return


def get_sampler_chain():

    with open("chain.pkl", "rb") as f:

        sampler_chain_array = joblib.load(f)

    return sampler_chain_array


def make_convergence_plot(sampler_chain, ndim, burnin):

    dimension1 = int(np.round(np.sqrt(ndim),0))

    if dimension1**2 < ndim:

        dimension2 = dimension1 + 1

    else:

        dimension2 = dimension1

    fig, axes = plt.subplots(nrows=dimension1, ncols=dimension2 , sharex=True, figsize=(dimension1*5.0, dimension2*2.0))

    axes = axes.ravel()

    for k in range(ndim):
        y = sampler_chain[:,:,k].copy()
        y = y[:, burnin:]

        # Compute the estimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(100), np.log(y.shape[1]), 20)).astype(int)
        gw2010 = np.empty(len(N))
        new = np.empty(len(N))
        for i, n in enumerate(N):
            gw2010[i] = autocorr_gw2010(y[:, :n])
            new[i] = autocorr_new(y[:, :n])


        axes[k].loglog(N, gw2010, "o-", label="G\&W 2010")
        axes[k].loglog(N, new, "o-", label="new")
        #ylim = plt.gca().get_ylim()
        axes[k].plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
        #axes[k].set_ylim(ylim)
        axes[k].set_xlabel("number of samples, $N$")
        axes[k].set_ylabel(r"$\tau$ estimates")
        axes[k].legend(fontsize=14)
        axes[k].set_title(str(k))

    return



def extract_best_fit_parameters(samples, variables, variables_order):
    # Print the samples array
    print(samples)

    # Calculate the percentiles of the parameter samples (lower 2.5%, lower 15.9%, median 50%, upper 84.1%, upper 97.5%) 
    # Extract them and create a list
    best = list(map(lambda v: [v[0],v[1],v[2],v[3],v[4]],  \
                            zip(*np.percentile(samples, [2.5, 15.9, 50, 84.1, 97.5], axis=0))))

    # Match varying variables with the corresponded best-fit value
    i = 0
    for p in variables_order:
        if variables[p]['vary']:
            variables[p]['best'] = best[i]
            i = i+1
            print(p, variables[p]['best'])

    # Return the best fit parameters and the updated dictionary
    return np.array(best), variables





def make_corner_plot(samples, best, variables, variable_order, ndim):

    variable_names = []
    for variable_name in variable_order:
        if variables[variable_name]['vary']:
            variable_names.append(variables[variable_name]['texname'])

    if samples.shape[1] > 1e5:

        samples = samples[::int(samples.shape[1]/1000)]



    fig, axes = plt.subplots(ndim, ndim, figsize=(12.5,9))
    corner(samples, bins=40, labels=variable_names,
                      max_n_ticks=3,plot_contours=True,quantiles=[0.025,0.975],fig=fig,
                      show_titles=True, title_quantiles =  [0.16,0.5,0.84], verbose=True,truths=best[:,2],range=np.ones(ndim)*1.0) # fix this
    fig.subplots_adjust(bottom=0.1,left=0.1,top=0.8)

    return


def get_fitting_region(wave, flux, error, wave_lims_list):

    assert type(wave_lims_list) is list, "%r is not a list" % wave_lims_list

    fit_mask = (wave >= wave_lims_list[0][0]) & (wave <= wave_lims_list[0][1])

    return wave[fit_mask], flux[fit_mask], error[fit_mask]


def get_continuum_region(wave, flux, error, wave_lims_list):

    assert type(wave_lims_list) is list, "%r is not a list" % wave_lims_list

    mask = np.zeros(len(wave),dtype=bool)

    for i in range(len(wave_lims_list)):

        mask += (wave >= wave_lims_list[i][0]) & (wave <= wave_lims_list[i][1])

    return wave[mask], flux[mask], error[mask]


def make_parameter_dictionary(variables_order):

    assert type(variables_order) is list, "%r is not a list" % variables_order


    oneparam = {'texname':'', 'vary': True, 'value':1., 
            'scale': 1., 'min':0., 'max':1.,
            'best': np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            'Gaussian prior': False, 'prior mean': 0, 'prior stddev': 0}


    variables = {}

    for variable_name in variables_order:

        variables.update({variable_name: copy.deepcopy(oneparam)})

    return variables


def perform_variable_check(variables):

    for p in variables:
      if not variables[p]['vary']:
        if not (variables[p]['value'] >= variables[p]['min']) & (variables[p]['value'] <= variables[p]['max']):
            raise ValueError("The fixed variables (" + p + ") need to be within the variable's min-max range")












