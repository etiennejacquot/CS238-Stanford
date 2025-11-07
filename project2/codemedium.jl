using CSV, DataFrames, Random, Statistics, Dates

# Q-learning
function batch_q_learning(df; num_states, num_actions, gamma, alpha, epochs=50)
    Q = 0.001 * randn(num_states, num_actions) 

    for epoch in 1:epochs
        delta = 0.0
        for row in eachrow(shuffle(df))
            s, a, r, sp = row.s, row.a, row.r, row.sp

            # reward shaping
            r += 0.1

            # terminal state handling 
            if sp == s || r != 0
                target = r
            else
                target = r + gamma * maximum(Q[sp, :])
            end

            target = clamp(target, -1000.0, 1000.0)

            old = Q[s, a]
            update = alpha * (target - old)
            Q[s, a] += update

            delta = max(delta, abs(Q[s, a] - old))
        end
        println(" $epoch delta=$delta")
    end
    return Q
end

# Policy extraction
function extract_policy(Q)
    num_states, _ = size(Q)
    policy = [argmax(Q[s, :]) for s in 1:num_states]
    return policy
end

# main 
function main()

    case = ARGS[1]
    if case == "small"
        file, gamma, alpha = "small.csv", 0.95, 0.05
        num_states, num_actions = 100, 4
    elseif case == "medium"
        file, gamma, alpha = "medium.csv", 0.95, 0.01
        num_states, num_actions = 50000, 7
    elseif case == "large"
        file, gamma, alpha = "large.csv", 0.95, 0.05
        num_states, num_actions = 302020, 9
    end

    println("loading $file")
    df = CSV.read(file, DataFrame)

    println("Running Q-learning ...")
    t_start = now()
    Q = batch_q_learning(df; num_states=num_states, num_actions=num_actions, gamma=gamma, alpha=alpha)
    t_end = now()
    elapsed = (t_end - t_start)
    println("Training time: $(Dates.value(elapsed) / 1000) seconds")

    println("Extracting policy ...")
    policy = extract_policy(Q)

    open("$(case).policy", "w") do f
        for a in policy
            println(f, a)
        end
    end

    println("unique(policy) = ", unique(policy))
    println("mean(Q) = ", mean(Q), " max(Q) = ", maximum(Q))
    println("Saved policy to $(case).policy")
end
main()
