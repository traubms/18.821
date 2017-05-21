import sympy 
from asymmetric_processes import State
def extract_a_b_terms(exp):
    a, b = sympy.Symbol('a'), sympy.Symbol('b')
    terms, syms = exp.as_terms()
    if a in syms:
        ia = syms.index(a)
    else:
        ia = -1
    if b in syms:
        ib = syms.index(b)
    else:
        ib = -1
    results = []
    for term in terms:
        coef = term[1][0][0]
        pows = term[1][1]
        if ia > -1:
            ac = pows[ia]
        else:
            ac = 0
        if ib > -1:
            bc = pows[ib]
        else:
            bc = 0
        results += [(coef, ac, bc)]
    return results

def pop_max_terms(terms, isA):
    c, a, b = zip(*terms)
    if isA:
        m = max(a)
        popped = [terms[i] for i in range(len(terms)) if a[i] == m]
        remaining = [terms[i] for i in range(len(terms)) if a[i] != m]
    else:
        m = max(b)
        popped = [terms[i] for i in range(len(terms)) if b[i] == m]
        remaining = [terms[i] for i in range(len(terms)) if b[i] != m]
    return m, popped, remaining
    
def get_groups(exp):
    terms = extract_a_b_terms(exp)
    As = dict()
    Bs = dict()
    c = 0
    isA = True
    while len(terms) > 0:
        isA = c % 4 in [0, 3]
        m, group, terms = pop_max_terms(terms, isA)
        if isA:
            As[m] = group
        else:
            Bs[m] = group
        c += 1
    return As, Bs

def main():
    N = 5
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    memo = dict([('0'*i + '1'*(N-i), b**i * a**(N-i))  for i in range(N+1)])
    queue = []
    queue = [State(s, cycle=False) for s in memo]
    print get_groups(a**5*b**4 + a**5*b**3 + a**4*b**5 + a**4*b**4 + a**3*b**5)

if __name__ == "__main__":
    main()