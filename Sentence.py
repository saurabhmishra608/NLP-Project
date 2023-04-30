from wordsegment import load, segment

load()
def getSentence(word):
  swap_out = []
  s = ""
  for i in range(len(word)):
    if word[i].isnumeric():
      swap_out.append((i,word[i]))
    else:
      s = s + word[i]

  # print(s)
  # print(swap_out)

  word = segment(s)
  r = ""
  for i in range(len(word)):
    r += word[i]
    if i != len(word)-1:
      r += " "
  word = r
  word += " "
  #print(word)
  j = 0
  i = 0
  k = 0
  ret = ""
  flag = 0
  while j < len(s) + len(swap_out):
    if i < len(swap_out) and j == swap_out[i][0]:
      ret = ret + swap_out[i][1]
      i += 1
      flag = 1
    else:
      if flag:
        flag = 0
        ret += " "
      ret += word[k]
      k += 1
      if k < len(word) and word[k] == " ":
        ret += word[k]
        k += 1
    j += 1
  if ret[-1] == " ":
    return ret[:-1]
  return ret
