cd "$(dirname "$0")"
cat <<'LIST' | xargs -P 4 -I{} sh -c "python3 twins.py {} > /dev/null"
--name laplace_p0 --p_order 0.0 --family laplace
--name graph_p0 --p_order 0.0 --graph
--name int32_p0 --p_order 0.0 --n_int 32
--name int128_p0 --p_order 0.0 --n_int 128
--name obs_p0 --p_order 0.0 --seed 1
--name obs_p1 --p_order 1.0 --seed 1
--name laplace_p0 --p_order 0.0 --family laplace --seed 1
--name int32_p0 --p_order 0.0 --n_int 32 --seed 1
LIST
echo DONE
