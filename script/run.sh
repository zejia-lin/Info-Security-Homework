
filepath=$1

filename=$(basename -- "$filepath")
extension="${filename##*.}"
filename="${filename%.*}"
shift;
touch ../build/${filename}
rm ../build/${filename} && sh ../script/bwm.sh ${filepath} && ../build/${filename}  "$@"