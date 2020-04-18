export const copyToClipboard = (valuee) => {
  let dummy = document.createElement("input");

  document.body.appendChild(dummy);
  dummy.value = valuee;
  dummy.select();
  document.execCommand("copy");
  document.body.removeChild(dummy);
}