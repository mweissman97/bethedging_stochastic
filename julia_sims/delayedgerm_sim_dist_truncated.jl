#code to calculate normalized probability of fixation curves incorporating delayed germination

using Distributions

# ============ ARGS =============
#stochstic/deterministic fitness
fits = parse(Bool, ARGS[1])
ftype = fits ? "fs" : "fd"
#stochstic/deterministic germination
germs = parse(Bool, ARGS[2])
gtype = germs ? "gs" : "gd"
#soft/hard selection
hards = parse(Bool, ARGS[3])
stype = hards ? "hs" : "ss"
#germination probabilities for BH and WT
pGermBH = parse(Float64, ARGS[4])
pGermWT = parse(Float64, ARGS[5])
#seed bank length (number of generations seeds are allowed to stay dormant for before being discarded)
bankLength = parse(Int64, ARGS[6])
#fitness dist params
fmean = parse(Float64, ARGS[7])
fvar = parse(Float64, ARGS[8])
#site label
site = ARGS[9]
#output file
filename = "delayedgerm_$ftype:$gtype:$stype:$pGermBH:$pGermWT:$bankLength:$fmean:$fvar:$site.csv"
save_dir = "data/June29_truncated/$site" * "_dist/"
save_path = save_dir * filename

# ============ META PARAMS ============
#10^maxn is the largest population size
maxn = 5 
#number of population sizes to repeat
lengthn = 20 
#creates a vector of 20 population sizes that are evenly spaced on a log scale
a=range(0, stop=maxn, length=lengthn)
allN=[convert(Int64,ceil(10^i)) for i in a] 
allN = sort(allN)
#number of replicates
reps = floor(1000*10^maxn) 

# ============ FUNCTIONS ==============
"""
    stoch_fitness_select(mean, var)

Generate a random number from a truncated normal distribution.

# Arguments
- `mean`: The mean of the normal distribution.
- `var`: The variance of the normal distribution.

# Returns
- `w`: A random number generated from the truncated normal distribution, used as the fitness for the current generation.

"""
function stoch_fitness_select(mean, var)
    #truncating the normal distribution to [0, 2*mean]
    lower_bound = 0
    upper_bound = 2 * mean
    dist = Truncated(Normal(mean, sqrt(var)), lower_bound, upper_bound)
    w = rand(dist)
    return w
end


"""
    det_fitness_select(mean, var, generations)

Compute the deterministic fitness value based on the given mean, variance, and the index of the current generation.

# Arguments
- `mean`: The mean value.
- `var`: The variance value.
- `generations`: The number of generations (index).

# Returns
The computed deterministic fitness value bounded at zero.

"""
function det_fitness_select(mean, var, generations)
    w = 0.0
    #if the number of generations is odd, the fitness value is the mean plus the square root of the variance
    if isodd(generations)
        w = mean + sqrt(var)
    #if the number of generations is even, the fitness value is the mean minus the square root of the variance
    else
        w = mean - sqrt(var)
    end
    return w < 0 ? 0 : w #fitness cannot be negative
end


"""
    stoch_germ_select(seeds::Int64, pGerm::Float64)

This function simulates the germination process of seeds based on a given probability of germination.

# Arguments
- `seeds::Int64`: The total number of seeds.
- `pGerm::Float64`: The probability of germination for each seed.

# Returns
- `germinated::Int64`: The number of seeds that germinated.
- `non_germinated::Int64`: The number of seeds that did not germinate.

"""
function stoch_germ_select(seeds::Int64, pGerm::Float64)
    germinated = rand(Binomial(seeds, pGerm))
    return germinated, seeds-germinated
end


"""
    det_germ_select(seeds::Int64, pGerm::Float64)

This function calculates the number of germinated seeds and the number of ungerminated seeds based on the total number of seeds and the germination probability.

# Arguments
- `seeds::Int64`: The total number of seeds.
- `pGerm::Float64`: The probability of germination.

# Returns
- `germinated::Int64`: The number of germinated seeds.
- `ungerminated::Int64`: The number of ungerminated seeds.
"""
function det_germ_select(seeds::Int64, pGerm::Float64)
    germinated = round(seeds*pGerm)
    return germinated, seeds-germinated
end


"""
    hard_reproduction(PopNum, K, w)

Calculate the new population numbers after reproduction, taking into account the carrying capacity and the fitness of each population.

# Arguments
- `PopNum::Vector{Int}`: A vector containing the current population numbers of two populations.
- `K::Int`: The carrying capacity of the environment.
- `w::Float64`: The fitness of the first population relative to the second population.

# Returns
- `PopNum::Vector{Int}`: A vector containing the new population numbers after reproduction.

"""
function hard_reproduction(PopNum, K, w)
    #Check if the sum of the population is zero
    if sum(PopNum) == 0
        return [0, 0]
    end
    #Calculate new population numbers using the Poisson distribution
    lambdaBH = w * PopNum[1]
    lambdaWT = w * PopNum[2]
    PopNum = [rand(Poisson(lambdaBH)), rand(Poisson(lambdaWT))]
    #Check if the total population exceeds the carrying capacity
    if sum(PopNum) > K
        #Calculate proportion and ensure it is within [0,1]
        p = PopNum[1] / sum(PopNum)
        #Assign new population numbers based on binomial distribution
        PopNum[1] = rand(Binomial(K, p))
        PopNum[2] = K - PopNum[1]
    end
    return PopNum
end


"""
    soft_reproduction(PopNum, PopSize)

This function simulates the reproduction process of a population with two types of individuals: bet hedgers (BH) and wild type (WT). It takes in the current population numbers `PopNum` and the total population size `PopSize`. The function calculates the proportion of bet hedgers in the population and assigns new population numbers based on a binomial distribution.

# Arguments
- `PopNum`: A vector of length 2 representing the current population numbers of bet hedgers and wild type individuals, respectively.
- `PopSize`: The total population size.

# Returns
A vector of length 2 representing the number of offspring for bet hedgers and wild type individuals, respectively.

"""
function soft_reproduction(PopNum, PopSize)
    #Check if the sum of the population is zero
    if sum(PopNum) == 0
        return [0, 0]
    end
    #Calculate the proportion of bet hedgers
    p = PopNum[1] / sum(PopNum)
    #Assign new population numbers based on binomial distribution
    PopNum[1] = rand(Binomial(PopSize, p))
    PopNum[2] = PopSize - PopNum[1]
    #Returns vector of [# of BH offspring, # of WT offspring]
    return PopNum
end


"""
    germination(PopNum, BHbank, WTbank, pGermBH, pGermWT, germs)

Perform germination process on a population of seeds.

# Arguments
- `PopNum`: A vector representing the population numbers of two types of seeds (BH and WT).
- `BHbank`: A vector representing the seed bank of BH seeds.
- `WTbank`: A vector representing the seed bank of WT seeds.
- `pGermBH`: A scalar representing the germination rate of BH seeds.
- `pGermWT`: A scalar representing the germination rate of WT seeds.
- `germs`: A boolean indicating whether to use a stochastic germination function or a deterministic germination function.

# Returns
- A tuple containing the updated population numbers of BH and WT seeds, and the updated seed banks of BH and WT seeds.

# Details
- The function chooses a germination function based on the value of `germs`.
- It applies the germination function to the candidate seeds, including the active population and the seed bank.
- It separates the germinated and dormant seeds.
- It updates the population numbers with the germinated seeds.
- It deposits the dormant seeds back into the seed bank, shifting one generation and discarding the oldest generation.

"""
function germination(PopNum, BHbank, WTbank, pGermBH, pGermWT, germs)
    #choose the germination function
    fGerm = germs ? stoch_germ_select : det_germ_select
    #all candidate seeds, including the active population and the seed bank, length=bankLength+1
    BHcandidates = [PopNum[1]; BHbank]
    WTcandidates = [PopNum[2]; WTbank]
    #map germination function to the vector
    BHtuples = fGerm.(BHcandidates, pGermBH) #BH uses a germination rate
    WTtuples = fGerm.(WTcandidates, pGermWT) #WT has 100% germination (using the same function for consistency)
    #separate germinated and dormant seeds
    BHgermed = [t[1] for t in BHtuples]
    BHdormant = [t[2] for t in BHtuples]
    WTgermed = [t[1] for t in WTtuples]
    WTdormant = [t[2] for t in WTtuples]
    #update PopNum with germinated seeds
    PopNum[1] = sum(BHgermed)
    PopNum[2] = sum(WTgermed)
    #deposit dormant seeds back into the seed bank, shifting one generation and discarding the oldest generation
    BHbank = BHdormant[1:end-1]
    WTbank = WTdormant[1:end-1]
    #return updated PopNum, BHbank, WTbank
    return PopNum, BHbank, WTbank
end


"""
    generation(PopNum, K, generations, BHbank, WTbank, pGermBH, pGermWT, germs, fits, fmean, fvar)

Simulate a generation of the population dynamics, including fitness selection, reproduction, and germination processes.

# Arguments
- `PopNum`: The current population size.
- `K`: The carrying capacity.
- `generations`: The number of generations.
- `BHbank`: The bank of BH offspring.
- `WTbank`: The bank of WT offspring.
- `pGermBH`: The probability of germination for BH offspring.
- `pGermWT`: The probability of germination for WT offspring.
- `germs`: The germination rate.
- `fits`: A boolean indicating whether to use stochastic fitness selection or deterministic fitness selection.
- `fmean`: The mean fitness value.
- `fvar`: The fitness variance.

# Returns
- `PopNum`: The updated population size.
- `BHbank`: The updated bank of BH offspring.
- `WTbank`: The updated bank of WT offspring.

"""
function generation(PopNum, K, generations, BHbank, WTbank, pGermBH, pGermWT, germs, fits, fmean, fvar)
    #generation begins by determining the fitness
    w = fits ? stoch_fitness_select(fmean, fvar) : det_fitness_select(fmean, fvar, generations)
    #generate phenotypic distribution amongst diversified BH offspring
    PopNum, BHbank, WTbank = germination(PopNum, BHbank, WTbank, pGermBH, pGermWT, germs)
    # hard selection: use sum of curent population size; soft selection: use fixed carrying capacity
    PopNum = hards ? hard_reproduction(PopNum, K, w) : soft_reproduction(PopNum, K)
    #simulates random wright fisher reproduction
    return PopNum, BHbank, WTbank
end


"""
    simulate(PopSize::Int64, bankLength::Int64, fits::Bool, germs::Bool, fmean::Float64, fvar::Float64)

Simulates the evolution of two populations, BH (beneficial) and WT (wild type), until one population reaches fixation, both populations are lost, or 100*PopSize generations have passed.

# Arguments
- `PopSize::Int64`: The total population size.
- `bankLength::Int64`: The length of the BHbank and WTbank vectors.
- `fits::Bool`: A flag indicating whether fitness values are considered.
- `germs::Bool`: A flag indicating whether germination probabilities are considered.
- `fmean::Float64`: The mean fitness value.
- `fvar::Float64`: The variance of the fitness values.

# Returns
- `1`: BH fixation.
- `0`: BH loss.
- `-1`: Both populations are lost.
- `2`: Simulation ends after 100*PopSize generations.
"""
function simulate(PopSize::Int64, bankLength::Int64, fits::Bool, germs::Bool, fmean::Float64, fvar::Float64)
    # Function body
end
function simulate(PopSize::Int64, bankLength::Int64, fits::Bool, germs::Bool, fmean::Float64, fvar::Float64)
    InitNum = 1 #number of bh active to start
    BHbank = zeros(Int64, bankLength) #creates a vector of zeros of length bankLength
    WTbank = zeros(Int64, bankLength) #creates a vector of zeros of length bankLength
    PopNum = [InitNum, PopSize-InitNum] #PopNum = [# of BH active, # of WT active]
    generations::Int64 = 1
    #proceeds to the next generation until one population reaches fixation, both populations are lost, or 100*PopSize generations have passed
    while (0<PopNum[1]<PopSize || sum(BHbank)>0) && (0<PopNum[2]<PopSize || sum(WTbank)>0) && generations<PopSize*100
        PopNum, BHbank, WTbank = generation(PopNum, PopSize, generations, BHbank, WTbank, pGermBH, pGermWT, germs, fits, fmean, fvar)
        generations+=1
    end
    #determine the fixation outcome
    bhTot = PopNum[1] + sum(BHbank)
    wtTot = PopNum[2] + sum(WTbank)
    #1: BH fixation
    if PopNum[1] >= PopSize # bh reaches fixation
        return 1
    elseif bhTot > 0 && wtTot == 0 # wt is lost, bh is not lost
        return 1 
    #0: BH loss
    elseif PopNum[2] >= PopSize # wt reaches fixation
        return 0
    elseif bhTot == 0 && wtTot > 0 # bh is lost, wt is not lost
        return 0
    #-1: both populations are lost
    elseif bhTot == 0 && wtTot == 0
        return -1
    #2: simulation ends after 100*PopSize generations
    else
        return 2
    end
end

# ============ SIMULATION ============
println("ftype: ", ftype)
println("gtype: ", gtype)
println("stype: ", stype)
println("pGermBH: ", pGermBH)
println("pGermWT: ", pGermWT)
println("bankLength: ", bankLength)
println("fmean: ", fmean)
println("fvar: ", fvar)
println("site: ", site)
println("filename: ", filename)
colnames = ["N", "NPfix", "mutExt", "mutSurv"]
global out = open(save_path, "w") #creates a new output file whose filename includes parameters
write(out, join(colnames, ","), "\n")
close(out)
Npfix = Float64[] #creates an empty vector where normalized pfix values will be added
for N in allN #repeats this process at each population size
    c = 0 #counts number of replicates that reach fixation
    mutExt = 0 #counts number of replicates that result in mutual loss
    mutSurv = 0 #counts number of replicates that result in mutual survival
    for run = 1:reps
        fixation_outcome = simulate(N, bankLength, fits, germs, fmean, fvar) 
        # if mutual loss, increment mutExt
        if fixation_outcome == -1
            mutExt += 1
        # if mutual survival, increment mutSurv
        elseif fixation_outcome == 2
            mutSurv += 1
        # if fixation/loss, record the outcome
        else
            c += fixation_outcome
        end
    end
    # normalize with the number of replicates that result in only fixation/loss
    norm_pfix = (c / (reps - mutExt - mutSurv)) * N
    push!(Npfix, norm_pfix)
    output = [N, norm_pfix, mutExt, mutSurv]
    global out = open(save_path, "a") #adds the NPfix vector to the output file
    write(out, join(output, ","), "\n")
    close(out)
end
println(Npfix)

# documentation partially generated using GitHub Copilot