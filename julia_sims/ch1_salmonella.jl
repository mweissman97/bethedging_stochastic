using Distributions
using TickTock

#julia ch1_salmonella.jl $filename $pA $N $SLURM_ARRAY_TASK_ID

envs = true
etype = "e1"
phenos = true
ptype = "p1"

filename = join([ARGS[1], ARGS[4], "csv"], ".")
pA = parse(Float64,ARGS[2])
all_pSpec = [0.81]

println(filename)

#maxn = parse(Int64, ARGS[4])
#maxn = 9
#lengthn = 10
#a=range(0, stop=maxn, length = lengthn)
#allN_1=[convert(Int64,floor(10^i)) for i in a] #creates a vector of 10 population
#allN = allN_1[6:end]
#n_idx = parse(Int64,ARGS[3])
N = parse(Int64,ARGS[3])
reps = 100*N

tick()

function stoch_env_select(pA) #function that randomly generates the environment
    env = rand()
    return env <= pA
    #if returns true ==> in env. A; if returns false ==> in env. B
end

function det_env_select(generations, env_init)
    if env_init
        if isodd(generations)
            env = true
        else
            env = false
        end
    else
        if isodd(generations)
            env = false
        else
            env = true
        end
    end
end

function stoch_pheno(pSpec, BHcount)
    SpecPhe = rand(Binomial(BHcount, pSpec))
    #generates number of bet hedgers with specialist phenotype
    ConsPhe = BHcount-SpecPhe
    return [SpecPhe, ConsPhe]
    #returns a list with the # of bet hedgers with specialist and conservative phenotypes respectively
end

function det_pheno(pSpec, BHcount)
    SpecPhe = BHcount*pSpec
    #generates number of bet hedgers with specialist phenotype
    ConsPhe = BHcount-SpecPhe
    return [SpecPhe, ConsPhe]
    #returns a list with the # of bet hedgers with specialist and conservative phenotypes respectively
end

function average_fitness(PopNum, wBH, wWT)
    return (PopNum[1]*wBH+PopNum[2]*wWT)/(sum(PopNum))
end

function reproduction(env, PopNum, PopSize, spec, cons, phenotype_counts)
    #simulates random wright fisher reproduction in one generation
    wBH = Float64 #bet hedger fitness
    wWT = Float64 #wild type specialist fitness
    if env #collapse if else statement
        #if in env. A
        wBH = phenotype_counts[1]/PopNum[1]*spec[1]+phenotype_counts[2]/PopNum[1]*cons[1]
        wWT = spec[1]
        #wWT = cons[1]
    else
        #else in env. B
        wBH = phenotype_counts[1]/PopNum[1]*spec[2]+phenotype_counts[2]/PopNum[1]*cons[2]
        wWT = spec[2]
        #wWT = cons[2]
    end
    wBar = average_fitness(PopNum, wBH, wWT)
    p = PopNum[1]/PopSize * wBH/wBar
    PopNum[1] = rand(Binomial(PopSize, p))
    #number of bet hedger offspring is pulled randomly from a binomial distribution whose
    #expectation is based on the relative fitness of the bet hedger
    PopNum[2] = PopSize-PopNum[1]
    return PopNum
end

function generation(PopNum, spec, cons, pA, env_init, pSpec, generations, envs, phenos)
    if envs
        env = stoch_env_select(pA) #determines the environment for this generation
    else
        env = det_env_select(generations, env_init)
    end
    if phenos
        phenotype_counts = stoch_pheno(pSpec, PopNum[1]) #determines the realized phenotypic makeup of the BH
    else
        phenotype_counts = det_pheno(pSpec, PopNum[1])
    end
    PopNum = reproduction(env, PopNum, sum(PopNum), spec, cons, phenotype_counts)
    #simulates random wright fisher reproduction
    return PopNum
end

function simulate(PopSize::Int64, pA::Float64, pSpec::Float64, envs::Bool, phenos::Bool)
    InitNum = 1 #number of bh to start
    PopNum = [InitNum, PopSize-InitNum]
    #PopNum = [# of BH, # of WT]
    spec = [1.8, 0.003] #specialist phenotype in Env. A and Env. B respectively
    cons = [1, 0.04] #conservative phenotype in Env. A and Env. B respectively
    generations::Int64 = 1
    env_init = convert(Bool, rand(Binomial(1,pA)))
    while 0<PopNum[1]<PopSize
        PopNum = generation(PopNum, spec, cons, pA, env_init, pSpec, generations, envs, phenos)
        generations+=1
    end
    return PopNum[1]==PopSize
    #returns True (1) if BH win, False (0) otherwise
end

colnames = ["N", "NPfix", "pA"]
out = open(filename, "w") #creates a new output file whose filename includes parameters
write(out, join(colnames, ","), "\n") #populates output file with vector of 10 population sizes
close(out)

for pSpec in all_pSpec
    c = 0 #counts number of replicates that reach fixation
    for run = 1:reps
        if peektimer() > 536400
            r = run-1
            println(((c/r)*N))
            println(r)
        end
        c += simulate(N, pA, pSpec, envs, phenos) #c increases by 1 for each rep that reaches fixation
    end
    npf = ((c/reps)*N)
    output = [N, npf, pA]
    out = open(filename, "a") #adds the NPfix vector to the output file
    write(out, join(output, ","), "\n")
    close(out)

    println(npf)
end

tock()
