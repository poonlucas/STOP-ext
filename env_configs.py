# queue configs (arrival, service, connection)
queue_configs = {
    0: [(0.2, 0.3, 1.), (0.1, 0.8, 1.)],  # fully connected case
    1: [(0.2, 0.3, 0.95), (0.1, 0.8, 0.5)],  # Figure 4: https://proceedings.allerton.csl.illinois.edu/media/files/0062.pdf
    2: [(0.2, 0.3, 0.7), (0.1, 0.8, 0.5)],  # harder version of the above, lower connectivity to queue 1
    3: [(0.05, 0.9, 1.), (0.01, 0.85, 1.), (0.2, 0.95, 1.), (0.4, 0.75, 1.), (0.05, 0.9, 1.),
        (0.01, 0.9, 1.), (0.02, 0.85, 1.), (0.01, 0.9, 1.), (0.015, 0.9, 1.), (0.01, 0.85, 1.)]
}

# queue configs (arrival, service, connection)
ns_queue_configs = {
    0: [[(0.2, 0.3, 0.95), (0.1, 0.8, 0.5)], [(0.2, 0.3, 0.7), (0.1, 0.8, 0.5)]]
}

# criss-cross configs ([arrivals], [mus])
crisscross_configs = {
    0: ([0.3, 0.0, 0.3], [2., 1.5, 2.]),  # imbalanced low traffic (IL)
    1: ([0.6, 0.0, 0.6], [2., 1.5, 2.]),  # imbalanced medium traffic (IM)
    2: ([0.9, 0.0, 0.9], [2., 1.5, 2.]),  # imbalanced heavy traffic (IH)
    3: ([0.3, 0.0, 0.3], [2., 1., 2.]),  # balanced low traffic (BL)
    4: ([0.6, 0.0, 0.6], [2., 1., 2.]),  # balanced medium traffic (BM)
    5: ([0.9, 0.0, 0.9], [2., 1., 2.]),  # balanced high traffic (BH)
}
