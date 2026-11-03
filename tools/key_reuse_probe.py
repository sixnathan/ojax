import os
import re
import subprocess
import sys

os.environ["JAX_PLATFORMS"] = "cpu"


def reference():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_debug_key_reuse", True)
    from jax import random as R

    rows = {}

    def T(name, f):
        try:
            f()
            rows[name] = "OK"
        except Exception as e:
            rows[name] = "ERR " + str(e)
    k=lambda:R.key(0)
    def direct():
        a=k(); R.normal(a); R.normal(a)
    T("direct",direct)
    def split_then():
        a=k(); x,y=R.split(a); R.normal(x); R.normal(y)
    T("split",split_then)
    def jit_reuse():
        def f(a): return R.normal(a)+R.normal(a)
        jax.jit(f)(k())
    T("jit_reuse",jit_reuse)
    def jit_ok():
        def f(a):
            x,y=R.split(a); return R.normal(x)+R.normal(y)
        jax.jit(f)(k())
    T("jit_ok",jit_ok)
    def cond_each():
        def f(a,p): return jax.lax.cond(p,lambda a:R.normal(a),lambda a:R.uniform(a),a)
        jax.jit(f)(k(),True)
    T("cond_each_branch",cond_each)
    def cond_then_reuse():
        def f(a,p):
            x=jax.lax.cond(p,lambda a:R.normal(a),lambda a:0.0,a)
            return x+R.normal(a)
        jax.jit(f)(k(),True)
    T("cond_then_reuse",cond_then_reuse)
    def cond_one_side():
        def f(a,p): return jax.lax.cond(p,lambda a:R.normal(a),lambda a:0.0,a)
        jax.jit(f)(k(),True)
    T("cond_one_side",cond_one_side)
    def cond_close():
        a=k()
        def f(p): return jax.lax.cond(p,lambda:R.normal(a),lambda:R.uniform(a))
        jax.jit(f)(True)
    T("cond_closure",cond_close)
    def eager_cond_after():
        a=k(); jax.lax.cond(True,lambda a:R.normal(a),lambda a:0.0,a); R.normal(a)
    T("eager_cond_after",eager_cond_after)
    def scan_split():
        def body(c,_):
            c,s=R.split(c); return c,R.normal(s)
        jax.lax.scan(body,k(),None,length=3)
    T("scan_split",scan_split)
    def scan_reuse():
        def body(c,_): return c,R.normal(c)
        jax.lax.scan(body,k(),None,length=3)
    T("scan_reuse",scan_reuse)
    def scan_close():
        a=k()
        def body(c,_): return c,R.normal(a)
        jax.lax.scan(body,0,None,length=3)
    T("scan_closure",scan_close)
    def scan_xs():
        ks=R.split(k(),3)
        def body(c,x): return c,R.normal(x)
        jax.lax.scan(body,0,ks)
    T("scan_xs",scan_xs)
    def while_split():
        def c(s): return s[1]<3
        def b(s):
            a,i=s; a,_=R.split(a); return a,i+1
        jax.lax.while_loop(c,b,(k(),0))
    T("while_split",while_split)
    def while_reuse():
        def c(s): return s[1]<3
        def b(s):
            a,i=s; R.normal(a); return a,i+1
        jax.lax.while_loop(c,b,(k(),0))
    T("while_reuse",while_reuse)
    def while_close():
        a=k()
        def c(s): return s<3
        def b(s):
            R.normal(a); return s+1
        jax.lax.while_loop(c,b,0)
    T("while_closure",while_close)
    def fori_close():
        def b(i,s): return s
        jax.lax.fori_loop(0,3,b,k())
    T("fori_passthrough",fori_close)
    def switch_each():
        def f(a,i): return jax.lax.switch(i,[lambda a:R.normal(a),lambda a:R.uniform(a),lambda a:0.0],a)
        jax.jit(f)(k(),1)
    T("switch_each",switch_each)
    def vmap_split():
        jax.vmap(R.normal)(R.split(k(),3))
    T("vmap_split",vmap_split)
    def vmap_same():
        a=k()
        jax.vmap(lambda _:R.normal(a))(jnp.arange(3))
    T("vmap_same",vmap_same)
    def clone_ok():
        a=k(); R.normal(R.clone(a)); R.normal(a)
    T("clone_ok",clone_ok)
    def fold_in_twice():
        a=k(); R.fold_in(a,1); R.fold_in(a,2)
    T("fold_in_twice",fold_in_twice)
    def key_data_only():
        a=k(); R.key_data(a); R.normal(a)
    T("key_data_then",key_data_only)
    def nested_jit():
        def g(a): return R.normal(a)
        def f(a): return jax.jit(g)(a)+R.normal(a)
        jax.jit(f)(k())
    T("nested_jit_reuse",nested_jit)
    def nested_jit_ok():
        def g(a): return R.normal(a)
        def f(a):
            x,y=R.split(a); return jax.jit(g)(x)+jax.jit(g)(y)
        jax.jit(f)(k())
    T("nested_jit_ok",nested_jit_ok)
    def cond_return_key():
        def f(a,p): return jax.lax.cond(p,lambda a:a,lambda a:R.split(a)[0],a)
        jax.jit(f)(k(),True)
    T("cond_return_key",cond_return_key)
    def cond_return_key_reuse():
        def f(a,p):
            b=jax.lax.cond(p,lambda a:a,lambda a:a,a)
            return R.normal(a)+R.normal(b)
        jax.jit(f)(k(),True)
    T("cond_identity_alias_reuse",cond_return_key_reuse)
    def grad_ok():
        def f(x,a): return x*R.normal(a)
        jax.grad(f)(1.0,k())
    T("grad_ok",grad_ok)
    def grad_reuse():
        def f(x,a): return x*R.normal(a)*R.normal(a)
        jax.grad(f)(1.0,k())
    T("grad_reuse",grad_reuse)
    def scan_carry_two():
        def body(c,_):
            a,b=c; return (b,a),R.normal(a)
        jax.lax.scan(body,(k(),R.key(1)),None,length=2)
    T("scan_carry_swap",scan_carry_two)
    def raw_uint():
        a=R.PRNGKey(0); R.normal(a); R.normal(a)
    T("raw_uint_reuse",raw_uint)

    def cond_unused_then_use():
        def f(a,p):
            jax.lax.cond(p,lambda a:0.0,lambda a:1.0,a)
            return R.normal(a)
        jax.jit(f)(k(),True)
    T("cond_unused_then_use",cond_unused_then_use)
    def cond_forward_then_use():
        def f(a,p):
            b=jax.lax.cond(p,lambda a:a,lambda a:a,a)
            return R.normal(b)
        jax.jit(f)(k(),True)
    T("cond_forward_then_use",cond_forward_then_use)
    def cond_mixed_out():
        def f(a,p):
            b=jax.lax.cond(p,lambda a:a,lambda a:R.split(a)[0],a)
            return R.normal(b)
        jax.jit(f)(k(),True)
    T("cond_mixed_out",cond_mixed_out)
    def scan_pass_then_use():
        a=k()
        def body(c,_): return c,0.0
        out,_=jax.lax.scan(body,a,None,length=2)
        R.normal(out)
    T("scan_pass_then_use",scan_pass_then_use)
    def scan_pass_then_both():
        a=k()
        def body(c,_): return c,0.0
        out,_=jax.lax.scan(body,a,None,length=2)
        R.normal(out); R.normal(a)
    T("scan_pass_then_both",scan_pass_then_both)
    def scan_consumes_init_then_use():
        a=k()
        def body(c,_):
            c,s=R.split(c); return c,R.normal(s)
        jax.lax.scan(body,a,None,length=2)
        R.normal(a)
    T("scan_consumes_init_then_use",scan_consumes_init_then_use)
    def scan_out_usable():
        a=k()
        def body(c,_):
            c,s=R.split(c); return c,R.normal(s)
        out,_=jax.lax.scan(body,a,None,length=2)
        R.normal(out); 
    T("scan_out_usable",scan_out_usable)
    def while_pass_then_use():
        def c(s): return s[1]<3
        def b(s): return s[0],s[1]+1
        out,_=jax.lax.while_loop(c,b,(k(),0))
        R.normal(out)
    T("while_pass_then_use",while_pass_then_use)
    def while_cond_and_body():
        def c(s):
            return R.normal(s[0])<3
        def b(s):
            a,_=R.split(s[0]); return a,s[1]
        jax.lax.while_loop(c,b,(k(),0))
    T("while_cond_and_body",while_cond_and_body)
    def nested_identity_jit():
        def g(a): return a
        def f(a):
            b=jax.jit(g)(a); return R.normal(b)
        jax.jit(f)(k())
    T("nested_identity_jit",nested_identity_jit)
    def nested_unused_jit():
        def g(a): return 0.0
        def f(a):
            jax.jit(g)(a); return R.normal(a)
        jax.jit(f)(k())
    T("nested_unused_jit",nested_unused_jit)
    def cond_in_scan():
        def body(c,p):
            c,s=R.split(c)
            return c,jax.lax.cond(p,lambda s:R.normal(s),lambda s:0.0,s)
        jax.lax.scan(body,k(),jnp.array([True,False]))
    T("cond_in_scan",cond_in_scan)
    def scan_in_cond():
        def f(a,p):
            def inner(a):
                def body(c,_):
                    c,s=R.split(c); return c,R.normal(s)
                return jax.lax.scan(body,a,None,length=2)[1].sum()
            return jax.lax.cond(p,inner,lambda a:0.0,a)
        jax.jit(f)(k(),True)
    T("scan_in_cond",scan_in_cond)
    def cond_branch_reuse():
        def f(a,p): return jax.lax.cond(p,lambda a:R.normal(a)+R.normal(a),lambda a:0.0,a)
        jax.jit(f)(k(),True)
    T("cond_branch_reuse",cond_branch_reuse)
    def consumed_before_cond():
        a=k(); R.normal(a)
        jax.lax.cond(True,lambda a:R.normal(a),lambda a:0.0,a)
    T("consumed_before_cond",consumed_before_cond)
    def consumed_before_unused_cond():
        a=k(); R.normal(a)
        jax.lax.cond(True,lambda a:0.0,lambda a:0.0,a)
    T("consumed_before_unused_cond",consumed_before_unused_cond)
    def consumed_closure_in_branch():
        a=k(); R.normal(a)
        jax.lax.cond(True,lambda:R.normal(a),lambda:0.0)
    T("consumed_closure_in_branch",consumed_closure_in_branch)
    def closure_after_cond():
        a=k()
        jax.lax.cond(True,lambda:R.normal(a),lambda:0.0)
        R.normal(a)
    T("closure_after_cond",closure_after_cond)
    def map_split():
        jax.lax.map(R.normal,R.split(k(),2))
    T("map_split",map_split)

    return rows


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    ours = {}
    out = subprocess.run([sys.argv[1]], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.strip():
            name, _, rest = line.partition(" ")
            ours[name] = rest
    rows = reference()
    verdict = text = full = 0
    for name, ref in rows.items():
        if name not in ours:
            continue
        mine = ours[name]
        ref_cut = norm(ref.split("eqn=")[0])
        same_verdict = mine[:2] == ref[:2]
        mine_cut = mine.split("eqn=")[0] if "eqn=" in ref else mine
        same_text = norm(mine_cut) == ref_cut
        full += norm(mine) == norm(ref)
        if norm(mine) != norm(ref) and same_text:
            a, b = norm(mine), norm(ref)
            i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            print(name, "full text differs at", i, "JAX", repr(b[max(0, i - 60):i + 80]), "OJAX", repr(a[max(0, i - 60):i + 80]))
        verdict += same_verdict
        text += same_text
        if not (same_verdict and same_text):
            print(name)
            print(" JAX ", ref_cut[:200])
            print(" OJAX", norm(mine)[:200])
    print("verdict", verdict, "text up to the eqn dump", text, "full text", full, "of", sum(1 for n in rows if n in ours))


if __name__ == "__main__":
    main()
