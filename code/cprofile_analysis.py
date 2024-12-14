import pstats
p = pstats.Stats('cprofile-2.txt')
p.sort_stats('cumulative').print_stats(100)