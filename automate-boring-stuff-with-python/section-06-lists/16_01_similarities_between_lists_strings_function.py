########### Good Way ###########
def append_twice_good(a_list, val):
	a_list = a_list + [val, val]
	return a_list
	
nums = [1, 2, 3]
nums = append_twice_good(nums, 7)
print(nums)      # [1, 2, 3, 7, 7]

# nums --> 1 2 3
# --
# append_twice_good
# a_list --> 1 2 3 7 7 returns the value 
# val --> 7